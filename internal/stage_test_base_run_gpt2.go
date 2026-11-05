package internal

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/codecrafters-io/inference-engine-tester/internal/gpt2"
	"github.com/codecrafters-io/inference-engine-tester/internal/gpt2text"
	"github.com/codecrafters-io/tester-utils/test_case_harness"
)

// The prompt is "The capital of France is". GPT-2 continues it greedily with
// " the capital of the French Republic". Every one of these six steps has a gap
// of at least 0.33 between the top two logits, so float32 and float64 engines
// agree. Later steps have gaps as small as 0.007, which is why the stage stops
// at six.
var (
	gpt2Prompt   = []int{464, 3139, 286, 4881, 318}
	gpt2Expected = []int{262, 3139, 286, 262, 4141, 2066}
	gpt2Config   = gpt2.Config{NEmbd: 768, NHead: 12, NLayer: 12, NPositions: 1024, VocabSize: 50257}
)

func testRunGPT2(harness *test_case_harness.TestCaseHarness) error {
	weights, err := gpt2WeightsPath(harness)
	if err != nil {
		return err
	}
	workDir, err := os.MkdirTemp("", "inference-gpt2-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(workDir)
	if err := os.Symlink(weights, filepath.Join(workDir, "model.safetensors")); err != nil {
		return err
	}
	harness.Logger.Infof("Linked model.safetensors: GPT-2's weights (548 MB, 160 tensors)")

	prompt, err := gpt2text.Decode(gpt2Prompt)
	if err != nil {
		return err
	}
	files := []modelFile{
		{name: "config.json", data: jsonBytes(gpt2Config)},
		{name: "ids.json", data: jsonBytes(gpt2Prompt)},
		{name: "tokens.json", data: jsonBytes(map[string]any{"count": len(gpt2Expected)})},
	}
	for _, file := range files {
		if err := os.WriteFile(filepath.Join(workDir, file.name), file.data, 0o644); err != nil {
			return err
		}
		harness.Logger.Infof("Wrote %s: %s", file.name, file.data)
	}
	harness.Logger.Infof("The prompt %s is %q", jsonBytes(gpt2Prompt), prompt)

	args := []string{"generate", "model.safetensors", "config.json", "ids.json", "tokens.json"}
	harness.Logger.Infof("$ ./your_program.sh %s", joinArgs(args))
	executable := harness.NewExecutable()
	executable.WorkingDir = workDir
	executable.TimeoutInMilliseconds = int((300 * time.Second) / time.Millisecond)
	start := time.Now()
	result, err := executable.Run(args...)
	if err != nil {
		return fmt.Errorf("gpt2: %s", err.Error())
	}
	harness.Logger.Infof("Your program took %.1fs", time.Since(start).Seconds())
	if result.ExitCode != 0 && len(result.Stdout) == 0 {
		return fmt.Errorf("gpt2: your program exited with code %d and printed nothing to stdout", result.ExitCode)
	}
	got, err := parseJSON[[]int](result.Stdout)
	if err != nil {
		return fmt.Errorf("gpt2: %w", err)
	}
	if text, err := gpt2text.Decode(got); err == nil {
		harness.Logger.Infof("Your engine wrote: %q", text)
	}
	if message := diffInts(gpt2Expected, got); message != "" {
		return withHint("gpt2", message, gpt2Hint(got))
	}
	harness.Logger.Successf("Your engine matches GPT-2")
	return nil
}

// gpt2Hint points at the likely bug from where the ids first go wrong.
func gpt2Hint(got []int) string {
	expectedText, _ := gpt2text.Decode(gpt2Expected)
	if len(got) == 0 || got[0] != gpt2Expected[0] {
		return "GPT-2's first token is 262, \" the\". A wrong first token means the forward pass differs from GPT-2's. Check that you read n_head from config.json and skip the h.<layer>.attn.bias tensors."
	}
	return fmt.Sprintf("Your first token is right, so the forward pass works. GPT-2 writes %q. Check that each new token is appended to the sequence and uses its index in the whole sequence as its position.", expectedText)
}
