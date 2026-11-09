package internal

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
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
		var typeErr *json.UnmarshalTypeError
		if errors.As(err, &typeErr) {
			return zero, wrongShapeError(typeErr, reflect.TypeFor[T]())
		}
		return zero, notJSONError(trimmed, err)
	}
	var extra any
	if err := dec.Decode(&extra); err == nil {
		return zero, fmt.Errorf("expected one JSON value on stdout, got extra data after it. Send debug output to stderr")
	}
	return out, nil
}

// notJSONError shows the first line the program printed, since that line is
// usually a debug print or a value printed in the language's own syntax.
func notJSONError(stdout []byte, err error) error {
	line, _, _ := bytes.Cut(stdout, []byte("\n"))
	shown := string(line)
	if len(shown) > 120 {
		shown = shown[:120] + "..."
	}
	message := fmt.Sprintf("expected JSON on stdout, but your program printed %q, which isn't JSON (%v).\n"+
		"Print only the JSON result to stdout, and send debug output to stderr.", shown, err)
	if (bytes.HasPrefix(line, []byte("[")) || bytes.HasPrefix(line, []byte("{"))) && bytes.ContainsRune(line, '\'') {
		message += "\nJSON strings use double quotes, not single quotes. Encode the result with a JSON library, like json.dumps in Python."
	}
	return errors.New(message)
}

// wrongShapeError reports JSON that parsed but doesn't fit the result type,
// like a single row where the tester expects a list of rows.
func wrongShapeError(err *json.UnmarshalTypeError, want reflect.Type) error {
	return fmt.Errorf("expected %s on stdout, but your program's JSON has %s where the tester expects %s",
		describeJSONType(want), article(err.Value), describeJSONType(err.Type))
}

// describeJSONType names a Go result type the way the stage descriptions do.
func describeJSONType(t reflect.Type) string {
	switch t.Kind() {
	case reflect.Slice:
		return "an array of " + strings.TrimPrefix(strings.TrimPrefix(plural(t.Elem()), "a "), "an ")
	case reflect.Float32, reflect.Float64:
		return "a number"
	case reflect.Int, reflect.Int32, reflect.Int64:
		return "a whole number"
	default:
		return article(t.Kind().String())
	}
}

func plural(t reflect.Type) string {
	switch t.Kind() {
	case reflect.Slice:
		return "arrays of " + plural(t.Elem())
	case reflect.Float32, reflect.Float64:
		return "numbers"
	case reflect.Int, reflect.Int32, reflect.Int64:
		return "whole numbers"
	default:
		return t.Kind().String() + "s"
	}
}

// article puts "a" or "an" before a JSON value kind such as "string" or "object".
func article(noun string) string {
	if noun != "" && strings.ContainsRune("aeiou", rune(noun[0])) {
		return "an " + noun
	}
	return "a " + noun
}

func near(want, got, atol, rtol float64) bool {
	diff := math.Abs(want - got)
	return diff <= atol || diff <= rtol*math.Abs(want)
}

func diffMatrix(want, got [][]float64, atol, rtol float64) string {
	if len(want) != len(got) {
		return withExpectation(fmt.Sprintf("expected %d rows, got %d", len(want), len(got)), want, got)
	}
	for i := range want {
		if len(got[i]) != len(want[i]) {
			return withExpectation(fmt.Sprintf("row %d: expected %d values, got %d", i, len(want[i]), len(got[i])), want, got)
		}
		for j := range want[i] {
			if !near(want[i][j], got[i][j], atol, rtol) {
				return withExpectation(fmt.Sprintf("row %d, column %d: expected %v, got %v", i, j, want[i][j], got[i][j]), want, got)
			}
		}
	}
	return ""
}

func diffVector(want, got []float64, atol, rtol float64) string {
	if len(want) != len(got) {
		return withExpectation(fmt.Sprintf("expected %d values, got %d", len(want), len(got)), want, got)
	}
	for i := range want {
		if !near(want[i], got[i], atol, rtol) {
			return withExpectation(fmt.Sprintf("value %d: expected %v, got %v", i, want[i], got[i]), want, got)
		}
	}
	return ""
}

func diffInts(want, got []int) string {
	if len(want) != len(got) {
		return withExpectation(fmt.Sprintf("expected %d tokens, got %d", len(want), len(got)), want, got)
	}
	for i := range want {
		if want[i] != got[i] {
			return withExpectation(fmt.Sprintf("token %d: expected %d, got %d", i, want[i], got[i]), want, got)
		}
	}
	return ""
}

// withExpectation appends the whole expected and printed values to a
// difference. The first difference says where the output went wrong, and
// seeing both values whole is what shows why: a transposed result, a row of
// the right numbers in the wrong order, or an off-by-one all look alike when
// you can only see one element of each.
func withExpectation(message string, want, got any) string {
	return fmt.Sprintf("%s\nexpected: %s\n     got: %s", message, asJSON(want), asJSON(got))
}

// asJSON renders a value the way the stage descriptions write it, truncated
// so a large result cannot bury the difference above it.
func asJSON(value any) string {
	encoded, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprintf("%v", value)
	}
	if len(encoded) > 240 {
		return string(encoded[:240]) + "..."
	}
	return string(encoded)
}
