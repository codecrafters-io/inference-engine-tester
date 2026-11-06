package internal

import (
	"fmt"
	"strconv"
	"time"

	"github.com/codecrafters-io/inference-engine-tester/internal/gpt2"
	"github.com/codecrafters-io/tester-utils/test_case_harness"
)

func testAttention(harness *test_case_harness.TestCaseHarness) error {
	type attentionCase struct {
		name  string
		hint  string
		cfg   gpt2.Config
		layer int
		input [][]float64
		parts []namedTensor
	}
	layer1 := identityQKV(1, 1)
	layer1[3].values[0] = 5
	cases := []attentionCase{
		{
			name:  "causal",
			hint:  "Each position uses only the scores for itself and earlier positions. Drop the later scores before the softmax.",
			cfg:   gpt2.Config{NEmbd: 1, NHead: 1, NLayer: 1, NPositions: 4, VocabSize: 2},
			layer: 0,
			input: [][]float64{{1}, {2}},
			parts: identityQKV(0, 1),
		},
		{
			name:  "scale",
			hint:  "Divide each score by the square root of the head size, n_embd / n_head, before the softmax.",
			cfg:   gpt2.Config{NEmbd: 2, NHead: 1, NLayer: 1, NPositions: 4, VocabSize: 2},
			layer: 0,
			input: [][]float64{{1, 0}, {0, 1}},
			parts: identityQKV(0, 2),
		},
		{
			name:  "heads",
			hint:  "Split the query, key, and value into n_head heads, and run the softmax separately for each head.",
			cfg:   gpt2.Config{NEmbd: 2, NHead: 2, NLayer: 1, NPositions: 4, VocabSize: 2},
			layer: 0,
			input: [][]float64{{1, 0}, {0, 1}},
			parts: identityQKV(0, 2),
		},
		{
			name:  "layer",
			hint:  "Read the attention tensors for the layer number in the arguments, like h.1.attn.c_attn.weight.",
			cfg:   gpt2.Config{NEmbd: 1, NHead: 1, NLayer: 2, NPositions: 4, VocabSize: 2},
			layer: 1,
			input: [][]float64{{1}, {2}},
			parts: append(identityQKV(0, 1), layer1...),
		},
	}
	for _, testCase := range cases {
		file := mustModel(tensorsFrom(testCase.parts))
		weights, err := gpt2.Load(file)
		if err != nil {
			return err
		}
		expected, err := gpt2.Attention(weights, testCase.cfg, testCase.layer, testCase.input)
		if err != nil {
			return err
		}
		stdout, err := runModel(harness, testCase.name, []modelFile{
			{name: "model.safetensors", data: file},
			{name: "config.json", data: jsonBytes(testCase.cfg)},
			{name: "input.json", data: jsonBytes(testCase.input)},
		}, []string{"attn", "model.safetensors", "config.json", strconv.Itoa(testCase.layer), "input.json"}, nil, 15*time.Second)
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
