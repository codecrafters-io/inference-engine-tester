package internal

import (
	"fmt"
	"math/rand"
	"strconv"
	"time"

	"github.com/codecrafters-io/inference-engine-tester/internal/gpt2"
	"github.com/codecrafters-io/tester-utils/test_case_harness"
)

// matrixCase is one run of a command that reads rows from input.json and
// prints rows: attn and block.
type matrixCase struct {
	name  string
	hint  string
	cfg   gpt2.Config
	layer int
	input [][]float64
	parts []namedTensor
	// bugs are the mistakes the tester can name from the output. A random
	// case is only used once every one of them changes its answer.
	bugs []gpt2.Bug
}

type matrixOracle func(w *gpt2.Weights, cfg gpt2.Config, layer int, input [][]float64) ([][]float64, error)

func (c matrixCase) run(oracle matrixOracle, weights *gpt2.Weights) ([][]float64, error) {
	return oracle(weights, c.cfg, c.layer, c.input)
}

func (c matrixCase) mutants(oracle matrixOracle, weights *gpt2.Weights) []mutant[[][]float64] {
	return mutantsOf(weights, c.bugs, func(w *gpt2.Weights) ([][]float64, error) { return c.run(oracle, w) })
}

// randomMatrixCase fills in c's model and input, drawing until the case
// catches every bug in c.bugs.
func randomMatrixCase(rng *rand.Rand, oracle matrixOracle, c matrixCase, rows int, model func(*rand.Rand) []namedTensor) matrixCase {
	return settle(rng, c.name, func(rng *rand.Rand) (matrixCase, bool) {
		c.parts = model(rng)
		c.input = randomRows(rng, rows, c.cfg.NEmbd)
		_, weights := loadParts(c.parts)
		expected, err := c.run(oracle, weights)
		if err != nil {
			return c, false
		}
		return c, allCaught(c.bugs, c.mutants(oracle, weights), expected, matricesApart)
	})
}

func runMatrixCases(harness *test_case_harness.TestCaseHarness, command string, oracle matrixOracle, cases []matrixCase) error {
	for _, testCase := range cases {
		file, weights := loadParts(testCase.parts)
		expected, err := testCase.run(oracle, weights)
		if err != nil {
			return err
		}
		stdout, err := runModel(harness, testCase.name, []modelFile{
			{name: "model.safetensors", data: file},
			{name: "config.json", data: jsonBytes(testCase.cfg)},
			{name: "input.json", data: jsonBytes(testCase.input)},
		}, []string{command, "model.safetensors", "config.json", strconv.Itoa(testCase.layer), "input.json"}, nil, 15*time.Second)
		if err != nil {
			return err
		}
		got, err := parseJSON[[][]float64](stdout)
		if err != nil {
			return fmt.Errorf("%s: %w", testCase.name, err)
		}
		if message := diffMatrix(expected, got, 1e-4, 0); message != "" {
			hint := diagnose(got, testCase.mutants(oracle, weights), sameMatrix(1e-4, 0), testCase.hint)
			return withHint(testCase.name, message, hint)
		}
		harness.Logger.Successf("%s passed", testCase.name)
	}
	return nil
}

func bugList(lists ...[]gpt2.Bug) []gpt2.Bug {
	var out []gpt2.Bug
	for _, list := range lists {
		out = append(out, list...)
	}
	return out
}
