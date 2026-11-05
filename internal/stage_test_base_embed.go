package internal

import (
	"fmt"
	"time"

	"github.com/codecrafters-io/inference-engine-tester/internal/gpt2"
	"github.com/codecrafters-io/inference-engine-tester/internal/safetensors"
	"github.com/codecrafters-io/tester-utils/test_case_harness"
)

func testEmbeddings(harness *test_case_harness.TestCaseHarness) error {
	cases := []struct {
		name    string
		hint    string
		tensors []safetensors.Tensor
		ids     []int
	}{
		{
			name: "add",
			hint: "Add row id of wte.weight to row i of wpe.weight, where i is the id's index in ids.json.",
			tensors: []safetensors.Tensor{
				f32Tensor("wte.weight", []int{2, 2}, 1, 2, 3, 4),
				f32Tensor("wpe.weight", []int{2, 2}, 10, 20, 30, 40),
			},
			ids: []int{1, 0},
		},
		{
			name: "position",
			hint: "The same id at two positions gives two different rows. Add the position row for each index.",
			tensors: []safetensors.Tensor{
				f32Tensor("wte.weight", []int{2, 2}, 1, 2, 3, 4),
				f32Tensor("wpe.weight", []int{2, 2}, 10, 20, 30, 40),
			},
			ids: []int{0, 0},
		},
		{
			name: "lookup",
			hint: "Use each id to pick its row of wte.weight, and its index in ids.json to pick the row of wpe.weight.",
			tensors: []safetensors.Tensor{
				f32Tensor("wte.weight", []int{2, 2}, 1, 2, 3, 4),
				f32Tensor("wpe.weight", []int{3, 2}, 10, 20, 30, 40, 50, 60),
			},
			ids: []int{1, 1, 0},
		},
	}
	for _, testCase := range cases {
		file := mustModel(testCase.tensors)
		weights, err := gpt2.Load(file)
		if err != nil {
			return err
		}
		expected, err := gpt2.Embed(weights, testCase.ids)
		if err != nil {
			return err
		}
		stdout, err := runModel(harness, testCase.name, []modelFile{
			{name: "model.safetensors", data: file},
			{name: "ids.json", data: jsonBytes(testCase.ids)},
		}, []string{"embed", "model.safetensors", "ids.json"}, nil, 15*time.Second)
		if err != nil {
			return err
		}
		got, err := parseJSON[[][]float64](stdout)
		if err != nil {
			return fmt.Errorf("%s: %w", testCase.name, err)
		}
		if message := diffMatrix(expected, got, 1e-5, 0); message != "" {
			return withHint(testCase.name, message, testCase.hint)
		}
		harness.Logger.Successf("%s passed", testCase.name)
	}
	return nil
}
