package gpt2

// Bug is one mistake a learner's engine can make.
//
// The oracle can run with a single bug switched on. The tester uses that in
// two ways. When it builds a random model, it keeps drawing until every bug a
// stage should catch changes the answer by a wide margin, so a broken engine
// fails in the stage that introduces the bug. And when a learner's output is
// wrong, the tester compares it with what each bug prints, so the hint can
// name the step that went wrong.
type Bug int

const (
	NoBug Bug = iota

	// Embeddings.
	EmbedNoPosition   // token row only, no position row
	EmbedPositionByID // position row picked by token id
	EmbedTokenByIndex // token row picked by index in the sequence

	// Attention.
	AttnNoMask      // every position also reads later positions
	AttnNoScale     // scores not divided by sqrt(head size)
	AttnScaleByEmbd // scores divided by sqrt(n_embd)
	AttnOneHead     // n_head ignored: one head across the whole row
	AttnNoWeights   // the input row used as its own query, key, and value
	AttnNoBias      // c_attn.bias left out
	AttnSwapQK      // query and key read from each other's slice
	AttnNoProj      // output projection left out
	AttnNoProjBias  // c_proj.bias left out

	// Any command that takes a layer number.
	LayerZero // layer 0's tensors read whatever the layer number

	// Layer norm.
	LnNoAffine       // weight and bias left out
	LnSampleVariance // variance divided by count - 1

	// Block, first half.
	BlockNoLn1       // attention run on the rows as given
	BlockAddToNormed // attention's output added onto the normalized copy
	BlockNoResidual  // attention's output returned on its own

	// Block, second half.
	BlockNoLn2     // MLP run on the rows as they are
	BlockLn1ForMLP // ln_1's weights used before the MLP
	MlpFromInput   // MLP reads the block's input, not the row after attention
	MlpNoResidual  // MLP's output returned on its own
	MlpOntoNormed  // MLP's output added onto the normalized copy
	MlpNoGelu      // no function between the two multiplies
	MlpRelu        // ReLU in place of GELU
	MlpNoFcBias    // mlp.c_fc.bias left out
	MlpNoProjBias  // mlp.c_proj.bias left out

	// Logits.
	LogitsNoLnF    // final layer norm left out
	LogitsFirstRow // first position scored, not the last
	LogitsOneLayer // only layer 0 run
	PickTieHigher  // equal scores keep the higher id
	GenNoFeedback  // the first new id repeated, never fed back in

	// Sampling.
	SampleGreedy        // random numbers ignored
	SampleNoTemperature // logits not divided by temperature
	SampleNoTopP        // every token kept
	SampleTopPBefore    // the token that reaches top_p is dropped
	SampleNoRenorm      // kept probabilities not divided by their sum
	SampleIDOrder       // tokens walked in id order, not by probability
	SampleReuseRandom   // the first random number used for every token

	// KV cache.
	CachePositionZero       // every new token embedded at position 0
	CachePositionAmongNew   // positions counted from the first new token
	CacheSharedAcrossLayers // one list of keys and values for every layer
	CacheDropsPrompt        // the prompt's keys and values not kept

	bugCount
)

// AllBugs lists every bug, for tests that check each one is caught somewhere.
func AllBugs() []Bug {
	out := make([]Bug, 0, bugCount-1)
	for bug := NoBug + 1; bug < bugCount; bug++ {
		out = append(out, bug)
	}
	return out
}

// WithBug returns the same weights with one bug switched on.
func (w *Weights) WithBug(bug Bug) *Weights {
	return &Weights{tensors: w.tensors, bug: bug}
}
