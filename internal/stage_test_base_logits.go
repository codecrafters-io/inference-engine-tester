package internal

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/codecrafters-io/inference-engine-tester/internal/gpt2"
	"github.com/codecrafters-io/tester-utils/test_case_harness"
)

type logitsCase struct {
	name  string
	hint  string
	cfg   gpt2.Config
	ids   []int
	parts []namedTensor
	bugs  []gpt2.Bug
}

var logitsBugs = []gpt2.Bug{gpt2.LogitsNoLnF, gpt2.LogitsFirstRow, gpt2.EmbedNoPosition}

const logitsHint = "Embed the ids, run every layer from 0 through n_layer - 1, layer-normalize the last row with ln_f, then score token i with row i of wte.weight."

func (c logitsCase) mutants(weights *gpt2.Weights) []mutant[[]float64] {
	return mutantsOf(weights, c.bugs, func(w *gpt2.Weights) ([]float64, error) { return gpt2.Logits(w, c.cfg, c.ids) })
}

func randomLogitsCase(rng *rand.Rand, c logitsCase) logitsCase {
	return settle(rng, c.name, func(rng *rand.Rand) (logitsCase, bool) {
		c.parts = randomGPT(rng, c.cfg)
		c.ids = randomIDs(rng, 3, c.cfg.VocabSize)
		_, weights := loadParts(c.parts)
		expected, err := gpt2.Logits(weights, c.cfg, c.ids)
		if err != nil {
			return c, false
		}
		return c, allCaught(c.bugs, c.mutants(weights), expected, vectorsApart)
	})
}

func logitsCases(rng *rand.Rand) []logitsCase {
	return []logitsCase{
		// The stage description's worked example.
		{
			name:  "tied",
			hint:  "Layer-normalize the last row with ln_f, then score token i with row i of wte.weight.",
			cfg:   gpt2.Config{NEmbd: 2, NHead: 1, NLayer: 1, NPositions: 4, VocabSize: 2},
			ids:   []int{0},
			parts: logitParts(1, []float64{1, 1, 1, -1}, []float64{0, 0, 1, 0, 0, 0, 0, 0}, []float64{0, 0}, []float64{1, -1}),
			bugs:  logitsBugs,
		},
		randomLogitsCase(rng, logitsCase{
			name: "last",
			hint: logitsHint,
			cfg:  gpt2.Config{NEmbd: 4, NHead: 2, NLayer: 1, NPositions: 8, VocabSize: 6},
			bugs: logitsBugs,
		}),
		randomLogitsCase(rng, logitsCase{
			name: "layers",
			hint: logitsHint,
			cfg:  gpt2.Config{NEmbd: 4, NHead: 2, NLayer: 2, NPositions: 8, VocabSize: 6},
			bugs: bugList(logitsBugs, []gpt2.Bug{gpt2.LogitsOneLayer}),
		}),
	}
}

func testLogits(harness *test_case_harness.TestCaseHarness) error {
	for _, testCase := range logitsCases(newFixtureRand()) {
		file, weights := loadParts(testCase.parts)
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
			hint := diagnose(got, testCase.mutants(weights), sameVector(1e-3, 1e-3), testCase.hint)
			return withHint(testCase.name, message, hint)
		}
		harness.Logger.Successf("%s passed", testCase.name)
	}
	return nil
}

func logitParts(layers int, wte, wpe, lnW, lnB []float64) []namedTensor {
	parts := []namedTensor{
		{"wte.weight", []int{2, 2}, wte},
		{"wpe.weight", []int{4, 2}, wpe},
		{"ln_f.weight", []int{2}, lnW},
		{"ln_f.bias", []int{2}, lnB},
	}
	for layer := 0; layer < layers; layer++ {
		parts = append(parts, zeroLayer(layer, 2)...)
	}
	return parts
}
