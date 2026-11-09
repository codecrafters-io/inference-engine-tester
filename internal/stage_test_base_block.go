package internal

import (
	"fmt"
	"strconv"
	"time"

	"github.com/codecrafters-io/inference-engine-tester/internal/gpt2"
	"github.com/codecrafters-io/tester-utils/test_case_harness"
)

type blockCase struct {
	name  string
	hint  string
	layer int
	input [][]float64
	parts []namedTensor
}

// testResidual covers the first half of a block: the layer norm before
// attention, and adding attention's output onto the row it started from. The
// MLP tensors stay zero, so a learner can leave the MLP for a later stage.
func testResidual(harness *test_case_harness.TestCaseHarness) error {
	cfg := gpt2.Config{NEmbd: 2, NHead: 1, NLayer: 2, NPositions: 4, VocabSize: 2}
	cases := []blockCase{
		{
			name:  "residual",
			hint:  "Layer-normalize a copy of each row with ln_1, run attention on the copies, then add attention's output onto the row it started from, not onto the copy.",
			layer: 0,
			input: [][]float64{{1, 2}},
			parts: residualLayer(0, 2),
		},
		{
			name:  "rows",
			hint:  "Each row is normalized on its own numbers, and attention still uses only the rows before it.",
			layer: 0,
			input: [][]float64{{1, 2}, {3, 1}},
			parts: residualLayer(0, 2),
		},
		{
			name:  "layer",
			hint:  "Read the tensors for the layer number in the arguments, like h.1.ln_1.weight.",
			layer: 1,
			input: [][]float64{{1, 2}},
			parts: append(zeroLayer(0, 2), residualLayer(1, 2)...),
		},
	}
	return runBlockCases(harness, cfg, cases)
}

func testTransformerBlock(harness *test_case_harness.TestCaseHarness) error {
	cfg := gpt2.Config{NEmbd: 2, NHead: 1, NLayer: 2, NPositions: 4, VocabSize: 2}
	cases := []blockCase{
		{
			name:  "gelu",
			hint:  "Layer-normalize with ln_2, multiply by mlp.c_fc.weight, apply GELU with the tanh formula to every number, then multiply by mlp.c_proj.weight and add onto the row.",
			layer: 0,
			input: [][]float64{{1, 2}},
			parts: withMLPBump(0, 2),
		},
		{
			name:  "rows",
			hint:  "The MLP works on one row at a time, on that row's own numbers after ln_2.",
			layer: 0,
			input: [][]float64{{1, 2}, {3, 1}},
			parts: withMLPBump(0, 2),
		},
		{
			name:  "layer",
			hint:  "Read the block's tensors for the layer number in the arguments, like h.1.mlp.c_fc.weight.",
			layer: 1,
			input: [][]float64{{1, 2}},
			parts: append(zeroLayer(0, 2), withMLPBump(1, 2)...),
		},
	}
	return runBlockCases(harness, cfg, cases)
}

func runBlockCases(harness *test_case_harness.TestCaseHarness, cfg gpt2.Config, cases []blockCase) error {
	for _, testCase := range cases {
		file := mustModel(tensorsFrom(testCase.parts))
		weights, err := gpt2.Load(file)
		if err != nil {
			return err
		}
		expected, err := gpt2.Block(weights, cfg, testCase.layer, testCase.input)
		if err != nil {
			return err
		}
		stdout, err := runModel(harness, testCase.name, []modelFile{
			{name: "model.safetensors", data: file},
			{name: "config.json", data: jsonBytes(cfg)},
			{name: "input.json", data: jsonBytes(testCase.input)},
		}, []string{"block", "model.safetensors", "config.json", strconv.Itoa(testCase.layer), "input.json"}, nil, 15*time.Second)
		if err != nil {
			return err
		}
		got, err := parseJSON[[][]float64](stdout)
		if err != nil {
			return fmt.Errorf("%s: %w", testCase.name, err)
		}
		if message := diffMatrix(expected, got, 1e-4, 0); message != "" {
			return withHint(testCase.name, message, testCase.hint)
		}
		harness.Logger.Successf("%s passed", testCase.name)
	}
	return nil
}
