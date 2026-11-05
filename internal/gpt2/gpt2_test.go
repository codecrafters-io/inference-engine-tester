package gpt2

import (
	"math"
	"testing"

	"github.com/codecrafters-io/inference-engine-tester/internal/safetensors"
)

func TestCausalAttentionMatchesHandCalculation(t *testing.T) {
	// One head of size 1. Q, K, and V are the input itself.
	// Input rows [1] and [2]. Position 1 scores are 2 and 4.
	// softmax([2, 4]) weights the values 1 and 2.
	file := mustWrite(t, []safetensors.Tensor{
		vec("h.0.attn.c_attn.weight", []int{1, 3}, 1, 1, 1),
		vec("h.0.attn.c_attn.bias", []int{3}, 0, 0, 0),
		vec("h.0.attn.c_proj.weight", []int{1, 1}, 1),
		vec("h.0.attn.c_proj.bias", []int{1}, 0),
	})
	w, err := Load(file)
	if err != nil {
		t.Fatal(err)
	}
	got, err := Attention(w, Config{NEmbd: 1, NHead: 1}, 0, [][]float64{{1}, {2}})
	if err != nil {
		t.Fatal(err)
	}
	e2 := math.Exp(2)
	e4 := math.Exp(4)
	second := (e2*1 + e4*2) / (e2 + e4)
	assertClose(t, got[0][0], 1)
	assertClose(t, got[1][0], second)
}

func TestGreedyTieKeepsLowerID(t *testing.T) {
	// Identical token rows make every logit equal, so the lower id wins.
	file := mustWrite(t, tiedFile())
	w, err := Load(file)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{NEmbd: 2, NHead: 1, NLayer: 1, NPositions: 4, VocabSize: 2}
	got, err := Generate(w, cfg, []int{0}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0] != 0 || got[1] != 0 {
		t.Fatalf("expected [0 0], got %v", got)
	}
}

func TestCachedGenerateMatchesFullRecompute(t *testing.T) {
	file := mustWrite(t, winnerFile())
	w, err := Load(file)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{NEmbd: 2, NHead: 1, NLayer: 1, NPositions: 8, VocabSize: 2}
	got, err := Generate(w, cfg, []int{0}, 3)
	if err != nil {
		t.Fatal(err)
	}
	ids := []int{0}
	for range 3 {
		logits, err := Logits(w, cfg, ids)
		if err != nil {
			t.Fatal(err)
		}
		best := 0
		for i := 1; i < len(logits); i++ {
			if logits[i] > logits[best] {
				best = i
			}
		}
		ids = append(ids, best)
	}
	for i := range got {
		if got[i] != ids[i+1] {
			t.Fatalf("token %d: cache %d, recompute %d", i, got[i], ids[i+1])
		}
	}
}

func TestSampleKeepsSmallestTopPSet(t *testing.T) {
	// Logits are produced from a 1-token model whose two scores differ.
	// The hand check is on sampleToken itself, which is the rule the stage states.
	logits := []float64{math.Log(0.1), math.Log(0.2), math.Log(0.7)}
	probs := softmax(logits)
	if probs[2] < 0.69 || probs[2] > 0.71 {
		t.Fatalf("expected about 0.7, got %v", probs)
	}
	// top-p 0.75 keeps 0.7 and then 0.2. Random 0.95 lands in the second kept token
	// after renormalizing. Order by probability is 2, 1, 0.
	got := sampleToken(logits, 0.75, 0.95)
	if got != 1 {
		t.Fatalf("expected token 1, got %d", got)
	}
	// A random number inside the first kept token's renormalized mass stays there.
	// 0.7/0.9 ≈ 0.777, so 0.5 selects token 2.
	if sampleToken(logits, 0.75, 0.5) != 2 {
		t.Fatal("expected the top token")
	}
}

func TestSampleTieBreaksByLowerID(t *testing.T) {
	// Equal logits. Sorted order is 0 then 1. The first half of the line is token 0.
	if sampleToken([]float64{0, 0}, 1, 0.4) != 0 {
		t.Fatal("expected the lower id")
	}
	if sampleToken([]float64{0, 0}, 1, 0.6) != 1 {
		t.Fatal("expected the higher id once the first half is passed")
	}
}

func tiedFile() []safetensors.Tensor {
	return oneLayerFile(
		[]float64{1, 0, 1, 0}, // both token rows identical
		[]float64{0, 0, 0, 0},
	)
}

func winnerFile() []safetensors.Tensor {
	// Token 1's row points the same way as a positive residual, token 0 the other way.
	return oneLayerFile(
		[]float64{1, 0, 0, 1},
		[]float64{0, 0, 1, 0},
	)
}

func oneLayerFile(wte, wpe []float64) []safetensors.Tensor {
	zero := func(n int) []float64 {
		return make([]float64, n)
	}
	eye2 := []float64{1, 0, 0, 1}
	return []safetensors.Tensor{
		vec("wte.weight", []int{2, 2}, wte...),
		vec("wpe.weight", []int{8, 2}, append(wpe, zero(12)...)...),
		vec("ln_f.weight", []int{2}, 1, 1),
		vec("ln_f.bias", []int{2}, 0, 0),
		vec("h.0.ln_1.weight", []int{2}, 1, 1),
		vec("h.0.ln_1.bias", []int{2}, 0, 0),
		vec("h.0.attn.c_attn.weight", []int{2, 6}, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1),
		vec("h.0.attn.c_attn.bias", []int{6}, zero(6)...),
		vec("h.0.attn.c_proj.weight", []int{2, 2}, eye2...),
		vec("h.0.attn.c_proj.bias", []int{2}, 0, 0),
		vec("h.0.ln_2.weight", []int{2}, 1, 1),
		vec("h.0.ln_2.bias", []int{2}, 0, 0),
		vec("h.0.mlp.c_fc.weight", []int{2, 8}, zero(16)...),
		vec("h.0.mlp.c_fc.bias", []int{8}, zero(8)...),
		vec("h.0.mlp.c_proj.weight", []int{8, 2}, zero(16)...),
		vec("h.0.mlp.c_proj.bias", []int{2}, 0, 0),
	}
}

func vec(name string, shape []int, values ...float64) safetensors.Tensor {
	raw := make([]float32, len(values))
	for i, value := range values {
		raw[i] = float32(value)
	}
	return safetensors.Tensor{Name: name, DType: "F32", Shape: shape, Data: safetensors.Float32Data(raw...)}
}

func mustWrite(t *testing.T, tensors []safetensors.Tensor) []byte {
	t.Helper()
	file, err := safetensors.Write(tensors)
	if err != nil {
		t.Fatal(err)
	}
	return file
}

func assertClose(t *testing.T, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("got %v, want %v", got, want)
	}
}
