package internal

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/codecrafters-io/inference-engine-tester/internal/gpt2"
	"github.com/codecrafters-io/tester-utils/test_case_harness"
)

type embedCase struct {
	name  string
	hint  string
	parts []namedTensor
	ids   []int
	bugs  []gpt2.Bug
}

var embedBugs = []gpt2.Bug{gpt2.EmbedNoPosition, gpt2.EmbedPositionByID, gpt2.EmbedTokenByIndex}

const embedHint = "Add row id of wte.weight to row i of wpe.weight, where i is the id's index in ids.json."

func (c embedCase) mutants(weights *gpt2.Weights) []mutant[[][]float64] {
	return mutantsOf(weights, c.bugs, func(w *gpt2.Weights) ([][]float64, error) { return gpt2.Embed(w, c.ids) })
}

func embedCases(rng *rand.Rand) []embedCase {
	const vocab, positions, nEmbd = 5, 6, 4
	return []embedCase{
		// The stage description's worked example.
		{
			name: "add",
			hint: embedHint,
			parts: []namedTensor{
				{"wte.weight", []int{2, 2}, []float64{1, 2, 3, 4}},
				{"wpe.weight", []int{2, 2}, []float64{10, 20, 30, 40}},
			},
			ids:  []int{1, 0},
			bugs: embedBugs,
		},
		// Random tables, so the rows can't be known ahead of the run. The ids
		// repeat and are out of order, which is what tells a token id from a
		// position.
		settle(rng, "lookup", func(rng *rand.Rand) (embedCase, bool) {
			c := embedCase{
				name: "lookup",
				hint: embedHint,
				parts: []namedTensor{
					randomTensor(rng, "wte.weight", vocab, nEmbd),
					randomTensor(rng, "wpe.weight", positions, nEmbd),
				},
				ids:  randomIDs(rng, 4, vocab),
				bugs: embedBugs,
			}
			_, weights := loadParts(c.parts)
			expected, err := gpt2.Embed(weights, c.ids)
			if err != nil {
				return c, false
			}
			return c, allCaught(c.bugs, c.mutants(weights), expected, matricesApart)
		}),
	}
}

func testEmbeddings(harness *test_case_harness.TestCaseHarness) error {
	for _, testCase := range embedCases(newFixtureRand()) {
		file, weights := loadParts(testCase.parts)
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
			hint := diagnose(got, testCase.mutants(weights), sameMatrix(1e-5, 0), testCase.hint)
			return withHint(testCase.name, message, hint)
		}
		harness.Logger.Successf("%s passed", testCase.name)
	}
	return nil
}
