package internal

import "github.com/codecrafters-io/inference-engine-tester/internal/gpt2"

// bugHints is what the tester says when a learner's output matches what the
// oracle prints with that bug switched on. Each hint names the step that went
// wrong and what the step should do.
var bugHints = map[gpt2.Bug]string{
	gpt2.EmbedNoPosition:   "Your rows are the token rows alone. Add row i of wpe.weight to each one, where i is the id's index in ids.json.",
	gpt2.EmbedPositionByID: "You're picking the position row with the token id. Use the id's index in ids.json to pick the row of wpe.weight.",
	gpt2.EmbedTokenByIndex: "You're picking the token row with the index. Use each id itself to pick its row of wte.weight.",

	gpt2.AttnNoMask:      "Your output matches attention that also reads later rows. Each position uses only the scores for itself and earlier positions. Drop the later scores before the softmax.",
	gpt2.AttnNoScale:     "Your output matches attention without the division. Divide each score by the square root of the head size, n_embd / n_head, before the softmax.",
	gpt2.AttnScaleByEmbd: "You're dividing the scores by the square root of n_embd. Divide by the square root of the head size, n_embd / n_head.",
	gpt2.AttnOneHead:     "Your output matches one head across the whole row. Split the query, key, and value into n_head heads, and run the softmax separately for each head.",
	gpt2.AttnNoWeights:   "Your output matches attention that uses each input row as its own query, key, and value. Multiply each row by c_attn.weight and add c_attn.bias first, then split the result into the query, key, and value.",
	gpt2.AttnNoBias:      "Your output matches attention without c_attn.bias. Add it after multiplying each row by c_attn.weight.",
	gpt2.AttnSwapQK:      "Your query and key are swapped. The result of c_attn splits into the query, then the key, then the value.",
	gpt2.AttnNoProj:      "Your output matches attention before the output projection. Multiply the joined heads by c_proj.weight and add c_proj.bias.",
	gpt2.AttnNoProjBias:  "Your output matches attention without c_proj.bias. Add it after multiplying by c_proj.weight.",

	gpt2.LayerZero: "Your output matches layer 0's tensors. Read the tensors for the layer number in the arguments, like h.1.attn.c_attn.weight.",

	gpt2.LnNoAffine:       "Your layer norm leaves out its weight and bias. After dividing, multiply each number by the layer norm's weight and add its bias.",
	gpt2.LnSampleVariance: "Your variance divides by one less than the count. Divide the squared distances by the count of numbers in the row.",

	gpt2.BlockNoLn1:       "Your output matches attention run on the rows as given. Layer-normalize a copy of each row with ln_1 first, and run attention on the copies.",
	gpt2.BlockAddToNormed: "You're adding attention's output onto the normalized copy. Add it onto the row the block was given.",
	gpt2.BlockNoResidual:  "Your output is attention's output on its own. Add it onto the row the block was given.",

	gpt2.BlockNoLn2:     "Your MLP runs on the row as it is. Layer-normalize a copy of the row with ln_2 first, and run the MLP on the copy.",
	gpt2.BlockLn1ForMLP: "You're normalizing with ln_1 before the MLP. The second layer norm uses ln_2.weight and ln_2.bias.",
	gpt2.MlpFromInput:   "Your MLP reads the block's input. It reads the row after attention's output has been added.",
	gpt2.MlpNoResidual:  "Your output is the MLP's output on its own. Add it onto the row its copy came from.",
	gpt2.MlpOntoNormed:  "You're adding the MLP's output onto the normalized copy. Add it onto the row the copy came from.",
	gpt2.MlpNoGelu:      "Your output matches an MLP without GELU. Apply GELU to every number between the two multiplies.",
	gpt2.MlpRelu:        "Your output matches an MLP that sets negative numbers to 0. Use GELU's tanh formula instead.",
	gpt2.MlpNoFcBias:    "Your output matches an MLP without mlp.c_fc.bias. Add it after multiplying by mlp.c_fc.weight, before GELU.",
	gpt2.MlpNoProjBias:  "Your output matches an MLP without mlp.c_proj.bias. Add it after multiplying by mlp.c_proj.weight.",

	gpt2.LogitsNoLnF:    "Your scores skip the final layer norm. Layer-normalize the last row with ln_f before scoring it against wte.weight.",
	gpt2.LogitsFirstRow: "You're scoring the first position. Score the last row only.",
	gpt2.LogitsOneLayer: "Your scores match running layer 0 alone. Run every layer from 0 through n_layer - 1 before ln_f.",
	gpt2.PickTieHigher:  "When scores are equal, keep the lower id. Compare scores with a strict greater-than.",
	gpt2.GenNoFeedback:  "Every new id is the first one repeated. Append each new id to the sequence, and score the longer sequence to pick the next.",

	gpt2.SampleGreedy:        "Your ids match taking the highest score. When tokens.json has a temperature, pick each token with the next random number from stdin.",
	gpt2.SampleNoTemperature: "Your ids match sampling without temperature. Divide every logit by temperature before the softmax.",
	gpt2.SampleNoTopP:        "Your ids match sampling from every token. Keep only the shortest highest-probability prefix whose total reaches top_p.",
	gpt2.SampleTopPBefore:    "You stop keeping tokens one too early. The token that takes the total to top_p is kept.",
	gpt2.SampleNoRenorm:      "Your ids match walking the kept probabilities as they are. Divide each one by the sum of the kept ones first, so they add up to 1.",
	gpt2.SampleIDOrder:       "You're walking the tokens in id order. Sort them by probability, highest first, and walk that order.",
	gpt2.SampleReuseRandom:   "You're using the first random number for every token. Each new token takes the next number from stdin.",

	gpt2.CachePositionZero:       "Each new token is embedded at its index in the whole sequence, not at position 0. The cache does not change which row of wpe.weight you add.",
	gpt2.CachePositionAmongNew:   "You're counting positions from the first new token. A new token's position is its index in the whole sequence, prompt included.",
	gpt2.CacheSharedAcrossLayers: "Your ids match one cache shared by every layer. Keep separate keys and values for each layer and each head.",
	gpt2.CacheDropsPrompt:        "Run every prompt row through the blocks first, in order, and keep each row's key and value. The first new token scores against all of them.",
}
