package internal

import (
	"math/rand"

	"github.com/codecrafters-io/inference-engine-tester/internal/gpt2"
	"github.com/codecrafters-io/tester-utils/test_case_harness"
)

var (
	residualBugs = []gpt2.Bug{
		gpt2.BlockNoLn1, gpt2.BlockAddToNormed, gpt2.BlockNoResidual,
		gpt2.LnNoAffine, gpt2.LnSampleVariance,
	}
	mlpBugs = []gpt2.Bug{
		gpt2.BlockNoLn2, gpt2.BlockLn1ForMLP, gpt2.MlpFromInput,
		gpt2.MlpNoResidual, gpt2.MlpOntoNormed, gpt2.MlpNoGelu, gpt2.MlpRelu,
		gpt2.MlpNoFcBias, gpt2.MlpNoProjBias,
	}
)

const (
	residualHint = "Layer-normalize a copy of each row with ln_1, run attention on the copies, then add attention's output onto the row it started from, not onto the copy."
	mlpHint      = "After attention's output is added, layer-normalize a copy of each row with ln_2, multiply by mlp.c_fc.weight and add mlp.c_fc.bias, apply GELU with the tanh formula to every number, multiply by mlp.c_proj.weight and add mlp.c_proj.bias, and add the result onto the row the copy came from."
)

// blockConfig is the model the random block cases use. Rows of 4 numbers keep
// their shape through a layer norm. A row of 2 always normalizes to about
// [-1, 1] or [1, -1], which hides most of what the steps before it did.
var blockConfig = gpt2.Config{NEmbd: 4, NHead: 2, NLayer: 2, NPositions: 4, VocabSize: 2}

func twoLayers(layer func(rng *rand.Rand, layer, nEmbd int) []namedTensor) func(*rand.Rand) []namedTensor {
	return func(rng *rand.Rand) []namedTensor {
		return append(layer(rng, 0, blockConfig.NEmbd), layer(rng, 1, blockConfig.NEmbd)...)
	}
}

// residualCases cover the first half of a block: the layer norm before
// attention, and adding attention's output onto the row it started from. The
// MLP tensors stay zero, so a learner can leave the MLP for the next stage.
func residualCases(rng *rand.Rand) []matrixCase {
	exampleConfig := gpt2.Config{NEmbd: 2, NHead: 1, NLayer: 2, NPositions: 4, VocabSize: 2}
	return []matrixCase{
		// The stage description's worked example.
		{
			name:  "residual",
			hint:  residualHint,
			cfg:   exampleConfig,
			input: [][]float64{{1, 2}},
			parts: residualLayer(0, 2),
			bugs:  residualBugs,
		},
		randomMatrixCase(rng, gpt2.Block, matrixCase{
			name: "rows",
			hint: residualHint,
			cfg:  blockConfig,
			bugs: residualBugs,
		}, 3, twoLayers(randomResidualLayer)),
		randomMatrixCase(rng, gpt2.Block, matrixCase{
			name:  "layer",
			hint:  residualHint,
			cfg:   blockConfig,
			layer: 1,
			bugs:  bugList(residualBugs, []gpt2.Bug{gpt2.LayerZero}),
		}, 3, twoLayers(randomResidualLayer)),
	}
}

func transformerBlockCases(rng *rand.Rand) []matrixCase {
	exampleConfig := gpt2.Config{NEmbd: 2, NHead: 1, NLayer: 2, NPositions: 4, VocabSize: 2}
	return []matrixCase{
		// The stage description's worked example.
		{
			name:  "gelu",
			hint:  mlpHint,
			cfg:   exampleConfig,
			input: [][]float64{{1, 2}},
			parts: withMLPBump(0, 2),
			bugs:  bugList(residualBugs, mlpBugs),
		},
		randomMatrixCase(rng, gpt2.Block, matrixCase{
			name: "rows",
			hint: mlpHint,
			cfg:  blockConfig,
			bugs: bugList(residualBugs, mlpBugs),
		}, 3, twoLayers(randomLayer)),
		randomMatrixCase(rng, gpt2.Block, matrixCase{
			name:  "layer",
			hint:  mlpHint,
			cfg:   blockConfig,
			layer: 1,
			bugs:  bugList(residualBugs, mlpBugs, []gpt2.Bug{gpt2.LayerZero}),
		}, 3, twoLayers(randomLayer)),
	}
}

func testResidual(harness *test_case_harness.TestCaseHarness) error {
	return runMatrixCases(harness, "block", gpt2.Block, residualCases(newFixtureRand()))
}

func testTransformerBlock(harness *test_case_harness.TestCaseHarness) error {
	return runMatrixCases(harness, "block", gpt2.Block, transformerBlockCases(newFixtureRand()))
}
