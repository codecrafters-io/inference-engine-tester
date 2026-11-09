package internal

import (
	"slices"
	"testing"
	"time"

	"github.com/codecrafters-io/inference-engine-tester/internal/gpt2"
)

// TestKVCacheFixtureIDs pins the ids the KV cache stage's two cases expect.
// The stage description quotes them, so a change here means the description
// is now wrong.
func TestKVCacheFixtureIDs(t *testing.T) {
	cfg := gpt2.Config{NEmbd: 2, NHead: 1, NLayer: 1, NPositions: 8, VocabSize: 2}
	file := mustModel(tensorsFrom(copyGPT([]float64{1, 0, 0, 1}, kvCacheWPE)))
	weights, err := gpt2.Load(file)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		prompt []int
		want   []int
	}{
		{prompt: []int{0, 1, 0}, want: []int{0, 1, 1}},
		{prompt: []int{0, 1}, want: []int{1, 0, 1}},
	}
	for _, testCase := range cases {
		got, err := gpt2.Generate(weights, cfg, testCase.prompt, 3)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, testCase.want) {
			t.Fatalf("prompt %v: got %v, want %v", testCase.prompt, got, testCase.want)
		}
	}
}

// TestKVCacheFixtureUsesTheForwardPass checks that the ids depend on the
// blocks. The greedy and sampling stages flatten ln_f so that their ids come
// from ln_f.bias alone, which would make this stage's test pass without any
// attention at all.
func TestKVCacheFixtureUsesTheForwardPass(t *testing.T) {
	parts := copyGPT([]float64{1, 0, 0, 1}, kvCacheWPE)
	for _, part := range parts {
		if part.name == "ln_f.weight" && !slices.Equal(part.values, []float64{1, 1}) {
			t.Fatalf("ln_f.weight is %v, so the final layer norm discards the blocks' output", part.values)
		}
	}
	cfg := gpt2.Config{NEmbd: 2, NHead: 1, NLayer: 1, NPositions: 8, VocabSize: 2}
	file := mustModel(tensorsFrom(parts))
	weights, err := gpt2.Load(file)
	if err != nil {
		t.Fatal(err)
	}
	// Zeroing the attention output projection changes the ids only if the ids
	// depend on attention in the first place.
	broken := copyGPT([]float64{1, 0, 0, 1}, kvCacheWPE)
	for i := range broken {
		if broken[i].name == "h.0.attn.c_proj.weight" {
			broken[i].values = zeros(4)
		}
	}
	brokenWeights, err := gpt2.Load(mustModel(tensorsFrom(broken)))
	if err != nil {
		t.Fatal(err)
	}
	good, err := gpt2.Generate(weights, cfg, []int{0, 1, 0}, 3)
	if err != nil {
		t.Fatal(err)
	}
	bad, err := gpt2.Generate(brokenWeights, cfg, []int{0, 1, 0}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if slices.Equal(good, bad) {
		t.Fatalf("dropping attention gives the same ids %v, so the fixture does not test the forward pass", good)
	}
}

// TestRunGPT2TimeoutCoversDownloadAndProgram keeps the final stage's timeout
// ahead of the work it has to allow. Retrying the download must never leave
// too little time for a correct program, which would surface as a stage
// cutoff rather than as a download error.
func TestRunGPT2TimeoutCoversDownloadAndProgram(t *testing.T) {
	var stageTimeout time.Duration
	for _, testCase := range testerDefinition.TestCases {
		if testCase.Slug == "zr7" {
			stageTimeout = testCase.Timeout
		}
	}
	if stageTimeout == 0 {
		t.Fatal("the zr7 stage has no timeout")
	}
	needed := gpt2DownloadBudget + gpt2ProgramTimeout
	if needed >= stageTimeout {
		t.Fatalf("the download budget (%s) plus the program timeout (%s) is %s, which does not fit in the stage's %s", gpt2DownloadBudget, gpt2ProgramTimeout, needed, stageTimeout)
	}
}
