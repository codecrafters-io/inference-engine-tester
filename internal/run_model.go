package internal

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/codecrafters-io/inference-engine-tester/internal/safetensors"
	"github.com/codecrafters-io/tester-utils/test_case_harness"
)

type modelFile struct {
	name string
	data []byte
}

func f32Tensor(name string, shape []int, values ...float64) safetensors.Tensor {
	raw := make([]float32, len(values))
	for i, value := range values {
		raw[i] = float32(value)
	}
	return safetensors.Tensor{Name: name, DType: "F32", Shape: shape, Data: safetensors.Float32Data(raw...)}
}

func mustModel(tensors []safetensors.Tensor) []byte {
	file, err := safetensors.Write(tensors)
	if err != nil {
		panic(err)
	}
	return file
}

func jsonBytes(value any) []byte {
	raw, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	return raw
}

func runModel(harness *test_case_harness.TestCaseHarness, name string, files []modelFile, args []string, stdin []byte, timeout time.Duration) ([]byte, error) {
	workDir, err := os.MkdirTemp("", "inference-model-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(workDir)

	for _, file := range files {
		path := filepath.Join(workDir, file.name)
		if err := os.WriteFile(path, file.data, 0o644); err != nil {
			return nil, err
		}
		if file.name == "model.safetensors" {
			logModel(harness, file.data)
			continue
		}
		harness.Logger.Infof("Wrote %s: %s", file.name, previewFile(file))
	}
	if len(stdin) > 0 {
		harness.Logger.Infof("Stdin: %s", bytes.TrimSpace(stdin))
	}
	harness.Logger.Infof("$ ./your_program.sh %s", joinArgs(args))

	executable := harness.NewExecutable()
	executable.WorkingDir = workDir
	executable.TimeoutInMilliseconds = int(timeout / time.Millisecond)
	var stdout []byte
	var exitCode int
	if len(stdin) > 0 {
		runResult, err := executable.RunWithStdin(stdin, args...)
		if err != nil {
			return nil, fmt.Errorf("%s: %s", name, err.Error())
		}
		stdout = runResult.Stdout
		exitCode = runResult.ExitCode
	} else {
		runResult, err := executable.Run(args...)
		if err != nil {
			return nil, fmt.Errorf("%s: %s", name, err.Error())
		}
		stdout = runResult.Stdout
		exitCode = runResult.ExitCode
	}
	if exitCode != 0 && len(bytes.TrimSpace(stdout)) == 0 {
		return nil, fmt.Errorf("%s: your program exited with code %d and printed nothing to stdout", name, exitCode)
	}
	return stdout, nil
}

// logModel lists every tensor the tester wrote, with its values, because a
// learner can't read the binary file to debug a wrong answer.
func logModel(harness *test_case_harness.TestCaseHarness, file []byte) {
	lines, err := safetensors.Describe(file)
	if err != nil {
		harness.Logger.Infof("Wrote model.safetensors (%d bytes). Header: %s", len(file), headerPreview(file))
		return
	}
	harness.Logger.Infof("Wrote model.safetensors (%d bytes) with %d tensors:", len(file), len(lines))
	for _, line := range lines {
		harness.Logger.Infof("  %s", line)
	}
}

// withHint adds a hint about the likely bug to a failed comparison.
func withHint(name, message, hint string) error {
	if hint == "" {
		return fmt.Errorf("%s: %s", name, message)
	}
	return fmt.Errorf("%s: %s\nHint: %s", name, message, hint)
}

func previewFile(file modelFile) string {
	text := string(file.data)
	if len(text) > 240 {
		return text[:240] + "..."
	}
	return text
}

func joinArgs(args []string) string {
	out := ""
	for i, arg := range args {
		if i > 0 {
			out += " "
		}
		out += arg
	}
	return out
}

func parseJSON[T any](raw []byte) (T, error) {
	var zero T
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return zero, fmt.Errorf("expected JSON on stdout, but your program printed nothing")
	}
	dec := json.NewDecoder(bytes.NewReader(trimmed))
	dec.DisallowUnknownFields()
	var out T
	if err := dec.Decode(&out); err != nil {
		return zero, fmt.Errorf("expected JSON on stdout: %w", err)
	}
	var extra any
	if err := dec.Decode(&extra); err == nil {
		return zero, fmt.Errorf("expected one JSON value on stdout, got extra data after it. Send debug output to stderr")
	}
	return out, nil
}

func near(want, got, atol, rtol float64) bool {
	diff := math.Abs(want - got)
	return diff <= atol || diff <= rtol*math.Abs(want)
}

func diffMatrix(want, got [][]float64, atol, rtol float64) string {
	if len(want) != len(got) {
		return fmt.Sprintf("expected %d rows, got %d", len(want), len(got))
	}
	for i := range want {
		if len(got[i]) != len(want[i]) {
			return fmt.Sprintf("row %d: expected %d values, got %d", i, len(want[i]), len(got[i]))
		}
		for j := range want[i] {
			if !near(want[i][j], got[i][j], atol, rtol) {
				return fmt.Sprintf("row %d, column %d: expected %v, got %v", i, j, want[i][j], got[i][j])
			}
		}
	}
	return ""
}

func diffVector(want, got []float64, atol, rtol float64) string {
	if len(want) != len(got) {
		return fmt.Sprintf("expected %d values, got %d", len(want), len(got))
	}
	for i := range want {
		if !near(want[i], got[i], atol, rtol) {
			return fmt.Sprintf("value %d: expected %v, got %v", i, want[i], got[i])
		}
	}
	return ""
}

func diffInts(want, got []int) string {
	if len(want) != len(got) {
		return fmt.Sprintf("expected %d tokens, got %d", len(want), len(got))
	}
	for i := range want {
		if want[i] != got[i] {
			return fmt.Sprintf("token %d: expected %d, got %d", i, want[i], got[i])
		}
	}
	return ""
}
