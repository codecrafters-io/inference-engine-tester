package internal

import (
	"fmt"
	"time"

	"github.com/codecrafters-io/inference-engine-tester/internal/gpt2"
	"github.com/codecrafters-io/tester-utils/test_case_harness"
)

func testTransformerBlock(harness *test_case_harness.TestCaseHarness) error {
	cfg := gpt2.Config{NEmbd: 2, NHead: 1, NLayer: 2, NPositions: 4, VocabSize: 2}
	gelu := withGeluBump(0, zeroLayer(0, 2))
	layer1 := withGeluBump(1, append(zeroLayer(0, 2), zeroLayer(1, 2)...))
	cases := []struct {
		name  string
		hint  string
		layer int
		input [][]float64
		parts []namedTensor
	}{
		{name: "residual", hint: "Add the attention output onto the block's input, then add the MLP output onto that sum.", layer: 0, input: [][]float64{{1, 2}}, parts: zeroLayer(0, 2)},
		{name: "gelu", hint: "Apply GELU, using the tanh formula, to every number after mlp.c_fc, then multiply by mlp.c_proj.weight.", layer: 0, input: [][]float64{{1, 2}}, parts: gelu},
		{name: "layer", hint: "Read the block's tensors for the layer in layer.json, like h.1.mlp.c_fc.bias.", layer: 1, input: [][]float64{{1, 2}}, parts: layer1},
	}
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
			{name: "layer.json", data: jsonBytes(testCase.layer)},
			{name: "input.json", data: jsonBytes(testCase.input)},
		}, []string{"block", "model.safetensors", "config.json", "layer.json", "input.json"}, nil, 15*time.Second)
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
