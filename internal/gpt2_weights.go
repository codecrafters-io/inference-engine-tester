package internal

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/codecrafters-io/tester-utils/test_case_harness"
)

const (
	gpt2WeightsURL    = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
	gpt2WeightsSHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
	gpt2WeightsSize   = 548105171
)

// gpt2WeightsPath returns the path of GPT-2's real weights on this machine.
//
// CODECRAFTERS_GPT2_WEIGHTS_PATH points at a file the test environment already
// provides. Without it, the tester downloads the file once into the user cache
// directory and checks its SHA-256 before using it.
func gpt2WeightsPath(harness *test_case_harness.TestCaseHarness) (string, error) {
	if path := os.Getenv("CODECRAFTERS_GPT2_WEIGHTS_PATH"); path != "" {
		if _, err := os.Stat(path); err != nil {
			return "", fmt.Errorf("CODECRAFTERS_GPT2_WEIGHTS_PATH: %w", err)
		}
		return path, nil
	}
	cacheDir, err := os.UserCacheDir()
	if err != nil {
		cacheDir = os.TempDir()
	}
	path := filepath.Join(cacheDir, "codecrafters", "gpt2", "model.safetensors")
	if info, err := os.Stat(path); err == nil && info.Size() == gpt2WeightsSize {
		return path, nil
	}
	harness.Logger.Infof("Downloading GPT-2's weights (548 MB). This happens once.")
	if err := downloadGPT2Weights(path); err != nil {
		return "", fmt.Errorf("couldn't download GPT-2's weights: %w", err)
	}
	return path, nil
}

func downloadGPT2Weights(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	response, err := http.Get(gpt2WeightsURL)
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
	if _, err := io.Copy(io.MultiWriter(partial, hash), response.Body); err != nil {
		partial.Close()
		return err
	}
	if err := partial.Close(); err != nil {
		return err
	}
	if got := hex.EncodeToString(hash.Sum(nil)); got != gpt2WeightsSHA256 {
		return fmt.Errorf("SHA-256 is %s, want %s", got, gpt2WeightsSHA256)
	}
	return os.Rename(partial.Name(), path)
}
