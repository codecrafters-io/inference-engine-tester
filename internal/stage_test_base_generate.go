package internal

import (
	"bytes"
	"fmt"
	"math"
	"time"

	"github.com/codecrafters-io/inference-engine-tester/internal/gpt2"
	"github.com/codecrafters-io/tester-utils/test_case_harness"
)

func testGreedyDecode(harness *test_case_harness.TestCaseHarness) error {
	cases := []struct {
		name  string
		hint  string
		ids   []int
		count int
		parts []namedTensor
	}{
		{name: "winner", hint: "Append the id with the highest score, and repeat until you have count new ids.", ids: []int{0}, count: 2, parts: withFinal(copyGPT([]float64{1, 0, 0, 1}, zeros(4)), []float64{0, 1})},
		{name: "tie", hint: "When scores are equal, keep the lower id. Compare scores with a strict greater-than.", ids: []int{0}, count: 2, parts: copyGPT([]float64{1, 0, 1, 0}, zeros(4))},
	}
	cfg := gpt2.Config{NEmbd: 2, NHead: 1, NLayer: 1, NPositions: 8, VocabSize: 2}
	for _, testCase := range cases {
		if err := runGenerate(harness, testCase.name, testCase.hint, cfg, testCase.ids, map[string]any{"count": testCase.count}, nil, testCase.parts); err != nil {
			return err
		}
	}
	return nil
}

func testSampling(harness *test_case_harness.TestCaseHarness) error {
	cfg := gpt2.Config{NEmbd: 2, NHead: 1, NLayer: 1, NPositions: 8, VocabSize: 2}
	sharp := withFinal(copyGPT([]float64{1, 0, 0, 1}, zeros(4)), []float64{0, math.Log(4)})
	cases := []struct {
		name  string
		hint  string
		ids   []int
		spec  map[string]any
		stdin string
		parts []namedTensor
	}{
		{
			name:  "tie",
			hint:  "Equal probabilities stay in id order. Choose the first token whose running total is greater than the random number.",
			ids:   []int{0},
			spec:  map[string]any{"count": 1, "temperature": 1.0, "top_p": 1.0},
			stdin: "0.6\n",
			parts: copyGPT([]float64{1, 0, 1, 0}, zeros(4)),
		},
		{
			name:  "top-p",
			hint:  "Keep the shortest highest-probability prefix whose total reaches top_p, and divide by its sum before walking it.",
			ids:   []int{0},
			spec:  map[string]any{"count": 1, "temperature": 1.0, "top_p": 0.75},
			stdin: "0.9\n",
			parts: sharp,
		},
		{
			name:  "temperature",
			hint:  "Divide every logit by temperature before the softmax.",
			ids:   []int{0},
			spec:  map[string]any{"count": 1, "temperature": 0.1, "top_p": 1.0},
			stdin: "0.9\n",
			parts: sharp,
		},
	}
	for _, testCase := range cases {
		if err := runGenerate(harness, testCase.name, testCase.hint, cfg, testCase.ids, testCase.spec, []byte(testCase.stdin), testCase.parts); err != nil {
			return err
		}
	}
	return nil
}

// kvCacheWPE gives positions 0 to 4 different rows, so an engine that embeds a
// new token at the wrong position writes different ids. The earlier generate
// stages zero wpe.weight, which leaves position handling untested until the
// real weights in the final stage.
var kvCacheWPE = []float64{0, 0, -2, 1, 1, -1, 0, 2, 1, 0}

func testKVCache(harness *test_case_harness.TestCaseHarness) error {
	cfg := gpt2.Config{NEmbd: 2, NHead: 1, NLayer: 1, NPositions: 8, VocabSize: 2}
	// ln_f is left as identity here, unlike the greedy and sampling stages, so
	// the ids depend on the whole forward pass rather than on ln_f.bias alone.
	parts := copyGPT([]float64{1, 0, 0, 1}, kvCacheWPE)
	cases := []struct {
		name  string
		hint  string
		ids   []int
		count int
	}{
		{
			name:  "prompt",
			hint:  "Run every prompt row through the blocks first, in order, and keep each row's key and value. The first new token scores against all of them.",
			ids:   []int{0, 1, 0},
			count: 3,
		},
		{
			name:  "positions",
			hint:  "Each new token is embedded at its index in the whole sequence, not at position 0. The cache does not change which row of wpe.weight you add.",
			ids:   []int{0, 1},
			count: 3,
		},
	}
	for _, testCase := range cases {
		if err := runGenerate(harness, testCase.name, testCase.hint, cfg, testCase.ids, map[string]any{"count": testCase.count}, nil, parts); err != nil {
			return err
		}
	}
	return nil
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

func runGenerate(harness *test_case_harness.TestCaseHarness, name, hint string, cfg gpt2.Config, ids []int, spec map[string]any, stdin []byte, parts []namedTensor) error {
	file := mustModel(tensorsFrom(parts))
	weights, err := gpt2.Load(file)
	if err != nil {
		return err
	}
	count := int(spec["count"].(int))
	var expected []int
	if _, ok := spec["temperature"]; ok {
		random := parseRandom(stdin)
		expected, err = gpt2.Sample(weights, cfg, ids, count, spec["temperature"].(float64), spec["top_p"].(float64), random)
	} else {
		expected, err = gpt2.Generate(weights, cfg, ids, count)
	}
	if err != nil {
		return err
	}
	stdout, err := runModel(harness, name, []modelFile{
		{name: "model.safetensors", data: file},
		{name: "config.json", data: jsonBytes(cfg)},
		{name: "ids.json", data: jsonBytes(ids)},
		{name: "tokens.json", data: jsonBytes(spec)},
	}, []string{"generate", "model.safetensors", "config.json", "ids.json", "tokens.json"}, stdin, 15*time.Second)
	if err != nil {
		return err
	}
	got, err := parseJSON[[]int](stdout)
	if err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	if message := diffInts(expected, got); message != "" {
		return withHint(name, message, hint)
	}
	harness.Logger.Successf("%s passed", name)
	return nil
}

func parseRandom(stdin []byte) []float64 {
	var out []float64
	for _, field := range bytes.Fields(stdin) {
		var value float64
		fmt.Sscanf(string(field), "%f", &value)
		out = append(out, value)
	}
	return out
}
