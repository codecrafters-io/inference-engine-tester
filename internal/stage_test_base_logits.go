package internal

import (
	"fmt"
	"time"

	"github.com/codecrafters-io/inference-engine-tester/internal/gpt2"
	"github.com/codecrafters-io/tester-utils/test_case_harness"
)

func testLogits(harness *test_case_harness.TestCaseHarness) error {
	wte := []float64{1, 0, 0, 1}
	tiedWte := []float64{1, 1, 1, -1}
	wpe := []float64{0, 0, 1, 0, 0, 0, 0, 0}
	lastWpe := []float64{0, 0, -1, 1, 0, 0, 0, 0}
	cases := []struct {
		name  string
		hint  string
		cfg   gpt2.Config
		ids   []int
		parts []namedTensor
	}{
		{
			name:  "tied",
			hint:  "Layer-normalize the last row with ln_f, then score token i with row i of wte.weight.",
			cfg:   gpt2.Config{NEmbd: 2, NHead: 1, NLayer: 1, NPositions: 4, VocabSize: 2},
			ids:   []int{0},
			parts: logitParts(1, tiedWte, wpe, []float64{0, 0}, []float64{1, -1}, false),
		},
		{
			name:  "last",
			hint:  "Score only the last position, after layer-normalizing it with ln_f. The variance divides the squared distances by the count once, not twice.",
			cfg:   gpt2.Config{NEmbd: 2, NHead: 1, NLayer: 1, NPositions: 4, VocabSize: 2},
			ids:   []int{0, 0},
			parts: logitParts(1, wte, lastWpe, []float64{1, 1}, []float64{0, 0}, false),
		},
		{
			name:  "layers",
			hint:  "Run every layer from 0 through n_layer - 1 before ln_f.",
			cfg:   gpt2.Config{NEmbd: 2, NHead: 1, NLayer: 2, NPositions: 4, VocabSize: 2},
			ids:   []int{0},
			parts: logitParts(2, wte, wpe, []float64{1, 1}, []float64{0, 0}, true),
		},
	}
	for _, testCase := range cases {
		file := mustModel(tensorsFrom(testCase.parts))
		weights, err := gpt2.Load(file)
		if err != nil {
			return err
		}
		expected, err := gpt2.Logits(weights, testCase.cfg, testCase.ids)
		if err != nil {
			return err
		}
		stdout, err := runModel(harness, testCase.name, []modelFile{
			{name: "model.safetensors", data: file},
			{name: "config.json", data: jsonBytes(testCase.cfg)},
			{name: "ids.json", data: jsonBytes(testCase.ids)},
		}, []string{"logits", "model.safetensors", "config.json", "ids.json"}, nil, 15*time.Second)
		if err != nil {
			return err
		}
		got, err := parseJSON[[]float64](stdout)
		if err != nil {
			return fmt.Errorf("%s: %w", testCase.name, err)
		}
		if message := diffVector(expected, got, 1e-3, 1e-3); message != "" {
			return withHint(testCase.name, message, testCase.hint)
		}
		harness.Logger.Successf("%s passed", testCase.name)
	}
	return nil
}

func logitParts(layers int, wte, wpe, lnW, lnB []float64, bumpSecond bool) []namedTensor {
	parts := []namedTensor{
		{"wte.weight", []int{2, 2}, wte},
		{"wpe.weight", []int{4, 2}, wpe},
		{"ln_f.weight", []int{2}, lnW},
		{"ln_f.bias", []int{2}, lnB},
	}
	for layer := 0; layer < layers; layer++ {
		one := zeroLayer(layer, 2)
		if bumpSecond && layer == 1 {
			one = withGeluBump(layer, one)
			for i := range one {
				if one[i].name == "h.1.mlp.c_proj.weight" {
					one[i].values[0] = 0
					one[i].values[1] = 1
				}
			}
		}
		parts = append(parts, one...)
	}
	return parts
}
