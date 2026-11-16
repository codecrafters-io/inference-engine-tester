package internal

import (
	"fmt"
	"math/rand"
	"slices"
	"strconv"
	"time"

	"github.com/codecrafters-io/inference-engine-tester/internal/gpt2"
	"github.com/codecrafters-io/tester-utils/test_case_harness"
)

// generateCase is one run of the generate command. A nil random means greedy
// decoding, with no temperature in tokens.json and nothing on stdin.
type generateCase struct {
	name        string
	hint        string
	cfg         gpt2.Config
	ids         []int
	count       int
	temperature float64
	topP        float64
	random      []float64
	parts       []namedTensor
	bugs        []gpt2.Bug
}

func (c generateCase) steps(weights *gpt2.Weights) ([]gpt2.Step, error) {
	return gpt2.Steps(weights, c.cfg, c.ids, c.count, c.temperature, c.topP, c.random)
}

func (c generateCase) tokens(weights *gpt2.Weights) ([]int, error) {
	steps, err := c.steps(weights)
	if err != nil {
		return nil, err
	}
	out := make([]int, len(steps))
	for i, step := range steps {
		out[i] = step.Token
	}
	return out, nil
}

func (c generateCase) mutants(weights *gpt2.Weights) []mutant[[]int] {
	return mutantsOf(weights, c.bugs, c.tokens)
}

func (c generateCase) spec() map[string]any {
	if c.random == nil {
		return map[string]any{"count": c.count}
	}
	return map[string]any{"count": c.count, "temperature": c.temperature, "top_p": c.topP}
}

func (c generateCase) stdin() []byte {
	var out []byte
	for _, value := range c.random {
		out = strconv.AppendFloat(out, value, 'f', -1, 64)
		out = append(out, '\n')
	}
	return out
}

// A random case is only used when every pick in it is clear. The ids have to
// match exactly, so a pick that turns on the sixth decimal would fail a
// correct engine that computes in 32-bit floats. These margins are thousands
// of times wider than that.
const (
	// scoreMargin separates the two highest logits of a greedy pick, and a
	// sampling pick's running totals from top_p and from the random number.
	scoreMargin = 0.02
	// orderMargin separates probabilities next to each other in the sorted
	// order.
	orderMargin = 0.001
)

func (c generateCase) clearPicks(steps []gpt2.Step) bool {
	for i, step := range steps {
		if c.random == nil {
			if gpt2.GreedyMargin(step.Logits) < scoreMargin {
				return false
			}
			continue
		}
		order, boundary := gpt2.SampleMargins(step.Logits, c.temperature, c.topP, c.random[i])
		if order < orderMargin || boundary < scoreMargin {
			return false
		}
	}
	return true
}

// randomGenerateCase fills in c's model, prompt, and random numbers, drawing
// until every pick is clear, accept is happy with the ids, and every bug in
// c.bugs prints ids that differ from the right ones and from each other's.
func randomGenerateCase(rng *rand.Rand, c generateCase, accept func(ids []int) bool) generateCase {
	sampling := c.temperature != 0
	return settle(rng, c.name, func(rng *rand.Rand) (generateCase, bool) {
		c.parts = randomGPT(rng, c.cfg)
		c.ids = randomIDs(rng, 3, c.cfg.VocabSize)
		if sampling {
			c.random = make([]float64, c.count)
			for i := range c.random {
				c.random[i] = float64(5+rng.Intn(91)) / 100
			}
		}
		_, weights := loadParts(c.parts)
		steps, err := c.steps(weights)
		if err != nil || !c.clearPicks(steps) {
			return c, false
		}
		expected, _ := c.tokens(weights)
		if accept != nil && !accept(expected) {
			return c, false
		}
		mutants := c.mutants(weights)
		return c, allCaught(c.bugs, mutants, expected, idsApart) && tellApart(mutants, idsApart)
	})
}

// generateConfig is the model the random generate cases use: two layers and
// two heads, with room for a three-id prompt and the new tokens.
var generateConfig = gpt2.Config{NEmbd: 4, NHead: 2, NLayer: 2, NPositions: 8, VocabSize: 6}

var (
	greedyBugs = []gpt2.Bug{gpt2.GenNoFeedback}
	topPBugs   = []gpt2.Bug{
		gpt2.SampleGreedy, gpt2.SampleNoTopP, gpt2.SampleTopPBefore,
		gpt2.SampleNoRenorm, gpt2.SampleIDOrder, gpt2.SampleReuseRandom,
	}
	temperatureBugs = []gpt2.Bug{
		gpt2.SampleGreedy, gpt2.SampleNoTemperature, gpt2.SampleIDOrder, gpt2.SampleReuseRandom,
	}
	cacheBugs = []gpt2.Bug{
		gpt2.CachePositionZero, gpt2.CachePositionAmongNew,
		gpt2.CacheSharedAcrossLayers, gpt2.CacheDropsPrompt,
	}
)

const (
	greedyHint   = "Append the id with the highest score, and repeat until you have count new ids."
	samplingHint = "Divide the logits by temperature, take the softmax, sort the tokens by probability, keep the shortest prefix whose total reaches top_p, divide the kept probabilities by their sum, and pick the first token whose running total is greater than the random number."
	cacheHint    = "With the cache, each id has to come out the same as without it. Keep every row's key and value for each layer and each head, in order, and embed each new token at its index in the whole sequence."
)

func greedyCases(rng *rand.Rand) []generateCase {
	example := gpt2.Config{NEmbd: 2, NHead: 1, NLayer: 1, NPositions: 8, VocabSize: 2}
	return []generateCase{
		// The stage description's worked example.
		{
			name: "winner", hint: greedyHint, cfg: example, ids: []int{0}, count: 2,
			parts: withFinal(copyGPT([]float64{1, 0, 0, 1}, zeros(4)), []float64{0, 1}),
			bugs:  []gpt2.Bug{gpt2.PickTieHigher},
		},
		// Both token rows are the same, so every score is tied.
		{
			name: "tie", hint: "When scores are equal, keep the lower id. Compare scores with a strict greater-than.",
			cfg: example, ids: []int{0}, count: 2,
			parts: copyGPT([]float64{1, 0, 1, 0}, zeros(4)),
			bugs:  []gpt2.Bug{gpt2.PickTieHigher},
		},
		// The new ids are not all the same, so each one has to be fed back in
		// before the next is picked.
		randomGenerateCase(rng, generateCase{
			name: "feedback", hint: greedyHint, cfg: generateConfig, count: 4, bugs: greedyBugs,
		}, nil),
	}
}

func samplingCases(rng *rand.Rand) []generateCase {
	example := gpt2.Config{NEmbd: 2, NHead: 1, NLayer: 1, NPositions: 8, VocabSize: 2}
	return []generateCase{
		// The stage description's worked example: two equal probabilities.
		{
			name: "tie", hint: "Equal probabilities stay in id order. Choose the first token whose running total is greater than the random number.",
			cfg: example, ids: []int{0}, count: 1, temperature: 1, topP: 1, random: []float64{0.6},
			parts: copyGPT([]float64{1, 0, 1, 0}, zeros(4)),
			bugs:  bugList(topPBugs, []gpt2.Bug{gpt2.SampleNoTemperature}),
		},
		randomGenerateCase(rng, generateCase{
			name: "top-p", hint: samplingHint, cfg: generateConfig, count: 4,
			temperature: 1, topP: []float64{0.6, 0.7, 0.8}[rng.Intn(3)], bugs: topPBugs,
		}, nil),
		randomGenerateCase(rng, generateCase{
			name: "temperature", hint: samplingHint, cfg: generateConfig, count: 4,
			temperature: []float64{0.5, 2}[rng.Intn(2)], topP: 1, bugs: temperatureBugs,
		}, nil),
	}
}

// kvCacheWPE gives positions 0 to 4 different rows, so an engine that embeds a
// new token at the wrong position writes different ids.
var kvCacheWPE = []float64{0, 0, -2, 1, 1, -1, 0, 2, 1, 0}

func kvCacheCases(rng *rand.Rand) []generateCase {
	example := gpt2.Config{NEmbd: 2, NHead: 1, NLayer: 1, NPositions: 8, VocabSize: 2}
	// ln_f is left as identity here, unlike the greedy example, so the ids
	// depend on the whole forward pass rather than on ln_f.bias alone.
	parts := copyGPT([]float64{1, 0, 0, 1}, kvCacheWPE)
	return []generateCase{
		// The stage description's worked example.
		{name: "prompt", hint: cacheHint, cfg: example, ids: []int{0, 1, 0}, count: 3, parts: parts, bugs: cacheBugs},
		{name: "positions", hint: cacheHint, cfg: example, ids: []int{0, 1}, count: 3, parts: parts, bugs: cacheBugs},
		// Two layers, so each layer needs keys and values of its own.
		randomGenerateCase(rng, generateCase{
			name: "layers", hint: cacheHint, cfg: generateConfig, count: 4, bugs: cacheBugs,
		}, nil),
	}
}

func testGreedyDecode(harness *test_case_harness.TestCaseHarness) error {
	return runGenerateCases(harness, greedyCases(newFixtureRand()))
}

func testSampling(harness *test_case_harness.TestCaseHarness) error {
	return runGenerateCases(harness, samplingCases(newFixtureRand()))
}

func testKVCache(harness *test_case_harness.TestCaseHarness) error {
	return runGenerateCases(harness, kvCacheCases(newFixtureRand()))
}

func withFinal(parts []namedTensor, bias []float64) []namedTensor {
	for i := range parts {
		if parts[i].name == "ln_f.weight" {
			parts[i].values = []float64{0, 0}
		}
		if parts[i].name == "ln_f.bias" {
			parts[i].values = bias
		}
	}
	return parts
}

func copyGPT(wte, wpeHead []float64) []namedTensor {
	wpe := zeros(16)
	copy(wpe, wpeHead)
	parts := []namedTensor{
		{"wte.weight", []int{2, 2}, wte},
		{"wpe.weight", []int{8, 2}, wpe},
		{"ln_f.weight", []int{2}, []float64{1, 1}},
		{"ln_f.bias", []int{2}, []float64{0, 0}},
	}
	parts = append(parts, identityQKV(0, 2)...)
	parts = append(parts,
		namedTensor{"h.0.ln_1.weight", []int{2}, []float64{1, 1}},
		namedTensor{"h.0.ln_1.bias", []int{2}, []float64{0, 0}},
		namedTensor{"h.0.ln_2.weight", []int{2}, []float64{1, 1}},
		namedTensor{"h.0.ln_2.bias", []int{2}, []float64{0, 0}},
		namedTensor{"h.0.mlp.c_fc.weight", []int{2, 8}, zeros(16)},
		namedTensor{"h.0.mlp.c_fc.bias", []int{8}, zeros(8)},
		namedTensor{"h.0.mlp.c_proj.weight", []int{8, 2}, zeros(16)},
		namedTensor{"h.0.mlp.c_proj.bias", []int{2}, []float64{0, 0}},
	)
	return parts
}

func runGenerateCases(harness *test_case_harness.TestCaseHarness, cases []generateCase) error {
	for _, testCase := range cases {
		file, weights := loadParts(testCase.parts)
		expected, err := testCase.tokens(weights)
		if err != nil {
			return err
		}
		stdout, err := runModel(harness, testCase.name, []modelFile{
			{name: "model.safetensors", data: file},
			{name: "config.json", data: jsonBytes(testCase.cfg)},
			{name: "ids.json", data: jsonBytes(testCase.ids)},
			{name: "tokens.json", data: jsonBytes(testCase.spec())},
		}, []string{"generate", "model.safetensors", "config.json", "ids.json", "tokens.json"}, testCase.stdin(), 15*time.Second)
		if err != nil {
			return err
		}
		got, err := parseJSON[[]int](stdout)
		if err != nil {
			return fmt.Errorf("%s: %w", testCase.name, err)
		}
		if message := diffInts(expected, got); message != "" {
			hint := diagnose(got, testCase.mutants(weights), slices.Equal[[]int], testCase.hint)
			return withHint(testCase.name, message, hint)
		}
		harness.Logger.Successf("%s passed", testCase.name)
	}
	return nil
}
