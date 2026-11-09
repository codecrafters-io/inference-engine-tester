package internal

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"

	"github.com/codecrafters-io/inference-engine-tester/internal/safetensors"
	"github.com/codecrafters-io/tester-utils/test_case_harness"
)

type inspectCase struct {
	name    string
	tensors []safetensors.Tensor
	hint    string
}

func runInspect(harness *test_case_harness.TestCaseHarness, testCase inspectCase) error {
	file, err := safetensors.Write(testCase.tensors)
	if err != nil {
		return err
	}
	expected, err := safetensors.Inspect(file)
	if err != nil {
		return fmt.Errorf("fixture: %w", err)
	}

	workDir, err := os.MkdirTemp("", "inference-inspect-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(workDir)

	path := filepath.Join(workDir, "model.safetensors")
	if err := os.WriteFile(path, file, 0o644); err != nil {
		return err
	}
	logModel(harness, file)
	harness.Logger.Infof("$ ./your_program.sh inspect model.safetensors")

	executable := harness.NewExecutable()
	executable.WorkingDir = workDir
	executable.TimeoutInMilliseconds = 15 * 1000
	result, err := executable.Run("inspect", "model.safetensors")
	if err != nil {
		return fmt.Errorf("%s: %s", testCase.name, err.Error())
	}
	if result.ExitCode != 0 && len(bytes.TrimSpace(result.Stdout)) == 0 {
		return fmt.Errorf("%s: your program exited with code %d and printed nothing to stdout", testCase.name, result.ExitCode)
	}

	actual, err := parseSummaries(result.Stdout)
	if err != nil {
		return fmt.Errorf("%s: %w", testCase.name, err)
	}
	if message, ok := diffSummaries(expected, actual); ok {
		return withHint(testCase.name, message, testCase.hint)
	}
	harness.Logger.Successf("%s passed", testCase.name)
	return nil
}

func parseSummaries(raw []byte) ([]safetensors.Summary, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return nil, fmt.Errorf("expected a JSON array on stdout, like [{\"name\":\"scale\"}], but your program printed nothing")
	}
	dec := json.NewDecoder(bytes.NewReader(trimmed))
	dec.DisallowUnknownFields()
	var actual []safetensors.Summary
	if err := dec.Decode(&actual); err != nil {
		return nil, fmt.Errorf("expected a JSON array of tensors on stdout: %w", err)
	}
	var extra any
	if err := dec.Decode(&extra); err == nil {
		return nil, fmt.Errorf("expected one JSON value on stdout, got extra data after it. Send debug output to stderr")
	}
	return actual, nil
}

func diffSummaries(expected, actual []safetensors.Summary) (string, bool) {
	if len(expected) != len(actual) {
		return withExpectation(fmt.Sprintf("expected %d tensors, got %d", len(expected), len(actual)),
			summaryNames(expected), summaryNames(actual)), true
	}
	// Past the count, the lists line up, so the whole lists would repeat
	// themselves. Showing only the entry that differs keeps the difference in
	// view, which matters most on the real weights and their 160 tensors.
	for i := range expected {
		want, got := expected[i], actual[i]
		if want.Name != got.Name {
			return withExpectation(fmt.Sprintf("tensor %d: expected name %q, got %q", i, want.Name, got.Name), want, got), true
		}
		if want.DType != got.DType {
			return withExpectation(fmt.Sprintf("%s: expected dtype %s, got %s", want.Name, want.DType, got.DType), want, got), true
		}
		if !reflect.DeepEqual(want.Shape, got.Shape) {
			return withExpectation(fmt.Sprintf("%s: expected shape %v, got %v", want.Name, want.Shape, got.Shape), want, got), true
		}
		if len(want.First) != len(got.First) {
			return withExpectation(fmt.Sprintf("%s: expected %d values, got %d", want.Name, len(want.First), len(got.First)), want, got), true
		}
		for j := range want.First {
			if math.Abs(want.First[j]-got.First[j]) > 1e-6 {
				return withExpectation(fmt.Sprintf("%s: first value %d: expected %v, got %v", want.Name, j, want.First[j], got.First[j]), want, got), true
			}
		}
	}
	return "", false
}

// summaryNames lists tensor names, which is what a wrong count is usually
// about: a skipped tensor, or an extra one the file holds but the stage says
// to leave out.
func summaryNames(summaries []safetensors.Summary) []string {
	names := make([]string, len(summaries))
	for i, summary := range summaries {
		names[i] = summary.Name
	}
	return names
}

func headerPreview(file []byte) string {
	if len(file) < 8 {
		return "(too short)"
	}
	n := int(binary.LittleEndian.Uint64(file[:8]))
	if len(file) < 8+n {
		return "(truncated)"
	}
	header := string(file[8 : 8+n])
	if len(header) > 180 {
		return header[:180] + "..."
	}
	return header
}
