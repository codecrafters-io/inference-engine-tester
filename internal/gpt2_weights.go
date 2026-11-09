package internal

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/codecrafters-io/tester-utils/test_case_harness"
)

const (
	// gpt2WeightsURL is a public copy of Hugging Face's openai-community/gpt2
	// model.safetensors, in the same region as the test-runner cluster, so the
	// download stays inside GCP and takes seconds. The bucket is Terraform's,
	// in modules/storage/gpt2_weights_bucket.tf, and scripts/upload_gpt2_weights.sh
	// fills it. CODECRAFTERS_GPT2_WEIGHTS_URL overrides this, which is how a
	// staging run can read its own bucket instead.
	gpt2WeightsURL    = "https://storage.googleapis.com/codecrafters-gpt2-weights-production/model.safetensors"
	gpt2WeightsSHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
	gpt2WeightsSize   = 548105171

	// The stage allows 5 minutes for the learner's program, so the download has
	// to fit in what is left of the stage's timeout. This is the budget for all
	// attempts together, not for each one, and a download that cannot finish
	// inside it has to fail as a download rather than eat the program's time.
	gpt2DownloadBudget   = 90 * time.Second
	gpt2DownloadAttempts = 3
	gpt2DownloadBackoff  = 2 * time.Second
)

// gpt2WeightsPath returns the path of GPT-2's real weights on this machine.
// Only the Run GPT-2 stage calls it, so no other stage downloads the file.
//
// CODECRAFTERS_GPT2_WEIGHTS_PATH points at a file that is already on disk,
// which is useful when developing the tester. Without it, the tester keeps the
// file at a fixed path in the temp directory on the container's disk. A later
// run in the same container reuses it, and a new container downloads it again.
func gpt2WeightsPath(harness *test_case_harness.TestCaseHarness) (string, error) {
	if path := os.Getenv("CODECRAFTERS_GPT2_WEIGHTS_PATH"); path != "" {
		if _, err := os.Stat(path); err != nil {
			return "", fmt.Errorf("CODECRAFTERS_GPT2_WEIGHTS_PATH: %w", err)
		}
		return path, nil
	}
	path := filepath.Join(os.TempDir(), "codecrafters-gpt2", "model.safetensors")
	if info, err := os.Stat(path); err == nil && info.Size() == gpt2WeightsSize {
		return path, nil
	}
	harness.Logger.Infof("Downloading GPT-2's weights (548 MB)")
	start := time.Now()
	// A failed download is almost always the host having a bad minute, and it
	// would otherwise fail the learner's submission for something they didn't
	// do. Every attempt and backoff shares one deadline, so retrying can never
	// borrow time from the learner's program.
	deadline := time.Now().Add(gpt2DownloadBudget)
	err := fmt.Errorf("no attempt was made within the %s budget", gpt2DownloadBudget)
	for attempt := 1; attempt <= gpt2DownloadAttempts; attempt++ {
		remaining := time.Until(deadline)
		if remaining <= 0 {
			break
		}
		if err = downloadGPT2Weights(path, remaining); err == nil {
			break
		}
		backoff := time.Duration(attempt) * gpt2DownloadBackoff
		if attempt == gpt2DownloadAttempts || time.Until(deadline) <= backoff {
			break
		}
		harness.Logger.Infof("Download attempt %d failed (%s), retrying", attempt, err)
		time.Sleep(backoff)
	}
	if err != nil {
		return "", fmt.Errorf("couldn't download GPT-2's weights within %s: %w", gpt2DownloadBudget, err)
	}
	harness.Logger.Infof("Downloaded GPT-2's weights in %.1fs", time.Since(start).Seconds())
	return path, nil
}

// weightsURL is gpt2WeightsURL unless CODECRAFTERS_GPT2_WEIGHTS_URL overrides
// it, which lets the weights move to a bucket without a tester release.
func weightsURL() string {
	if url := os.Getenv("CODECRAFTERS_GPT2_WEIGHTS_URL"); url != "" {
		return url
	}
	return gpt2WeightsURL
}

// downloadGPT2Weights streams the file to disk, so it never sits in memory,
// and only moves it into place once its size and SHA-256 match. timeout caps
// the whole request, including reading the body.
func downloadGPT2Weights(path string, timeout time.Duration) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	url := weightsURL()
	client := &http.Client{Timeout: timeout}
	response, err := client.Get(url)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("%s returned %s", url, response.Status)
	}
	partial, err := os.CreateTemp(filepath.Dir(path), "model-*.partial")
	if err != nil {
		return err
	}
	defer os.Remove(partial.Name())
	hash := sha256.New()
	written, err := io.Copy(io.MultiWriter(partial, hash), response.Body)
	if closeErr := partial.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	if written != gpt2WeightsSize {
		return fmt.Errorf("downloaded %d bytes, want %d", written, gpt2WeightsSize)
	}
	if got := hex.EncodeToString(hash.Sum(nil)); got != gpt2WeightsSHA256 {
		return fmt.Errorf("SHA-256 is %s, want %s", got, gpt2WeightsSHA256)
	}
	// CreateTemp makes the file readable only by its owner. The learner's
	// program reads it through a symlink, so let every user read it.
	if err := os.Chmod(partial.Name(), 0o644); err != nil {
		return err
	}
	return os.Rename(partial.Name(), path)
}
