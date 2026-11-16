package internal

import (
	"math/rand"

	"github.com/codecrafters-io/inference-engine-tester/internal/gpt2"
	"github.com/codecrafters-io/tester-utils/test_case_harness"
)

// attentionBugs change the answer for any number of heads. headBugs only show
// with more than one head, where the head size is no longer n_embd.
var (
	attentionBugs = []gpt2.Bug{
		gpt2.AttnNoMask, gpt2.AttnNoScale, gpt2.AttnNoWeights, gpt2.AttnNoBias,
		gpt2.AttnSwapQK, gpt2.AttnNoProj, gpt2.AttnNoProjBias,
	}
	headBugs = []gpt2.Bug{gpt2.AttnOneHead, gpt2.AttnScaleByEmbd}
)

const attentionHint = "Multiply each row by c_attn.weight and add c_attn.bias, split the result into the query, key, and value, and take the softmax of the scores for the positions up to this one. Then multiply the weighted average of the values by c_proj.weight and add c_proj.bias."

func randomAttentionLayers(layers int) func(*rand.Rand) []namedTensor {
	return func(rng *rand.Rand) []namedTensor {
		var parts []namedTensor
		for layer := 0; layer < layers; layer++ {
			parts = append(parts, randomAttention(rng, layer, 4)...)
		}
		return parts
	}
}

// attentionCases cover attention with a single head, so n_head is always 1
// and the head size is n_embd. Heads have a stage of their own.
func attentionCases(rng *rand.Rand) []matrixCase {
	return []matrixCase{
		// The stage description's worked example: one head of size 1 that
		// copies each row into its query, key, and value.
		{
			name:  "causal",
			hint:  "Each position uses only the scores for itself and earlier positions. Drop the later scores before the softmax.",
			cfg:   gpt2.Config{NEmbd: 1, NHead: 1, NLayer: 1, NPositions: 4, VocabSize: 2},
			input: [][]float64{{1}, {2}},
			parts: identityQKV(0, 1),
			bugs:  attentionBugs,
		},
		randomMatrixCase(rng, gpt2.Attention, matrixCase{
			name: "one-head",
			hint: attentionHint,
			cfg:  gpt2.Config{NEmbd: 4, NHead: 1, NLayer: 1, NPositions: 4, VocabSize: 2},
			bugs: attentionBugs,
		}, 3, randomAttentionLayers(1)),
		randomMatrixCase(rng, gpt2.Attention, matrixCase{
			name:  "layer",
			hint:  attentionHint,
			cfg:   gpt2.Config{NEmbd: 4, NHead: 1, NLayer: 2, NPositions: 4, VocabSize: 2},
			layer: 1,
			bugs:  bugList(attentionBugs, []gpt2.Bug{gpt2.LayerZero}),
		}, 3, randomAttentionLayers(2)),
	}
}

const headsHint = "Split the query, key, and value into n_head equal slices. For each head, score its slices, divide by the square root of the head size, n_embd / n_head, and take the softmax and the weighted average on that head alone. Put the heads' results side by side before c_proj."

// headsCases cover splitting attention into heads. A one-head engine from the
// stage before fails every one of them.
func headsCases(rng *rand.Rand) []matrixCase {
	bugs := bugList(attentionBugs, headBugs)
	return []matrixCase{
		// The stage description's worked example: two heads of size 1 that
		// copy each row into its query, key, and value.
		{
			name:  "two-heads",
			hint:  headsHint,
			cfg:   gpt2.Config{NEmbd: 2, NHead: 2, NLayer: 1, NPositions: 4, VocabSize: 2},
			input: [][]float64{{1, 0}, {0, 1}},
			parts: identityQKV(0, 2),
			bugs:  bugs,
		},
		randomMatrixCase(rng, gpt2.Attention, matrixCase{
			name: "heads",
			hint: headsHint,
			cfg:  gpt2.Config{NEmbd: 4, NHead: 2, NLayer: 1, NPositions: 4, VocabSize: 2},
			bugs: bugs,
		}, 3, randomAttentionLayers(1)),
		randomMatrixCase(rng, gpt2.Attention, matrixCase{
			name:  "layer",
			hint:  headsHint,
			cfg:   gpt2.Config{NEmbd: 4, NHead: 2, NLayer: 2, NPositions: 4, VocabSize: 2},
			layer: 1,
			bugs:  bugList(bugs, []gpt2.Bug{gpt2.LayerZero}),
		}, 3, randomAttentionLayers(2)),
	}
}

func testAttention(harness *test_case_harness.TestCaseHarness) error {
	return runMatrixCases(harness, "attn", gpt2.Attention, attentionCases(newFixtureRand()))
}

func testAttentionHeads(harness *test_case_harness.TestCaseHarness) error {
	return runMatrixCases(harness, "attn", gpt2.Attention, headsCases(newFixtureRand()))
}
