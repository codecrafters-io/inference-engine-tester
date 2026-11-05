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
	if err := downloadGPT2Weights(path); err != nil {
		return "", fmt.Errorf("couldn't download GPT-2's weights: %w", err)
	}
	harness.Logger.Infof("Downloaded GPT-2's weights in %.1fs", time.Since(start).Seconds())
	return path, nil
}

// downloadGPT2Weights streams the file to disk, so it never sits in memory,
// and only moves it into place once its size and SHA-256 match.
func downloadGPT2Weights(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	client := &http.Client{Timeout: gpt2DownloadTimeout}
	response, err := client.Get(gpt2WeightsURL)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("%s returned %s", gpt2WeightsURL, response.Status)
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
