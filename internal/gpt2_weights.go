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
	// model.safetensors. Serve it from the same region as the test runners so
	// the download takes seconds.
	gpt2WeightsURL    = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
	gpt2WeightsSHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
	gpt2WeightsSize   = 548105171

	// The test-runner pod is killed after 10 minutes, so a stalled download
	// has to fail well before that.
	gpt2DownloadTimeout = 3 * time.Minute

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
	// do. Three tries still leave room under the stage's timeout.
	var err error
	for attempt := 1; attempt <= gpt2DownloadAttempts; attempt++ {
		if err = downloadGPT2Weights(path); err == nil {
			break
		}
		if attempt < gpt2DownloadAttempts {
			harness.Logger.Infof("Download attempt %d failed (%s), retrying", attempt, err)
			time.Sleep(time.Duration(attempt) * gpt2DownloadBackoff)
		}
	}
	if err != nil {
		return "", fmt.Errorf("couldn't download GPT-2's weights after %d attempts: %w", gpt2DownloadAttempts, err)
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
// and only moves it into place once its size and SHA-256 match.
func downloadGPT2Weights(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	url := weightsURL()
	client := &http.Client{Timeout: gpt2DownloadTimeout}
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
