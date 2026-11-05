package gpt2

import (
	"os"
	"testing"
)

// TestRealGPT2Greedy runs the oracle on GPT-2's real weights. It is skipped
// unless GPT2_WEIGHTS points at model.safetensors, because the file is 548 MB.
//
// The expected ids are GPT-2's widely reported greedy continuation of
// "The capital of France is": " the capital of the French Republic".
func TestRealGPT2Greedy(t *testing.T) {
	path := os.Getenv("GPT2_WEIGHTS")
	if path == "" {
		t.Skip("set GPT2_WEIGHTS to GPT-2's model.safetensors to run this test")
	}
	file, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	w, err := Load(file)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{NEmbd: 768, NHead: 12, NLayer: 12, NPositions: 1024, VocabSize: 50257}
	got, err := Generate(w, cfg, []int{464, 3139, 286, 4881, 318}, 6)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{262, 3139, 286, 262, 4141, 2066}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}
