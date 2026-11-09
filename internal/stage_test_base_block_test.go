package internal

import (
	"slices"
	"testing"

	"github.com/codecrafters-io/inference-engine-tester/internal/gpt2"
)

func blockOutput(t *testing.T, parts []namedTensor, layer int, input [][]float64) [][]float64 {
	t.Helper()
	cfg := gpt2.Config{NEmbd: 2, NHead: 1, NLayer: 2, NPositions: 4, VocabSize: 2}
	weights, err := gpt2.Load(mustModel(tensorsFrom(parts)))
	if err != nil {
		t.Fatal(err)
	}
	out, err := gpt2.Block(weights, cfg, layer, input)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func matricesClose(t *testing.T, got, want [][]float64) bool {
	t.Helper()
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		for j := range got[i] {
			if diff := got[i][j] - want[i][j]; diff > 1e-4 || diff < -1e-4 {
				return false
			}
		}
	}
	return true
}

// TestResidualFixtureIsNotTheIdentity is the guard that matters. zeroLayer
// zeroes every attention tensor, so a correct block returns its input and a
// program that echoes input.json passes. residualLayer must not do that.
func TestResidualFixtureIsNotTheIdentity(t *testing.T) {
	input := [][]float64{{1, 2}}
	if got := blockOutput(t, zeroLayer(0, 2), 0, input); !matricesClose(t, got, input) {
		t.Fatalf("zeroLayer no longer returns its input (%v), so this guard needs rewriting", got)
	}
	got := blockOutput(t, residualLayer(0, 2), 0, input)
	if matricesClose(t, got, input) {
		t.Fatalf("residualLayer returns its input %v, so echoing input.json would pass", got)
	}
}

// TestResidualFixtureCatchesTheCommonMistakes checks the three ways a learner
// gets the first half of a block wrong.
func TestResidualFixtureCatchesTheCommonMistakes(t *testing.T) {
	input := [][]float64{{1, 2}}
	correct := blockOutput(t, residualLayer(0, 2), 0, input)
	// Adding onto the normalized copy instead of the row: about [-2, 2].
	// Skipping ln_1 entirely: [2, 4]. Dropping the residual: about [-1, 1].
	for _, wrong := range [][][]float64{
		{{-2, 2}},
		{{2, 4}},
		{{-1, 1}},
	} {
		if matricesClose(t, correct, wrong) {
			t.Errorf("a wrong implementation producing %v matches the expected %v", wrong, correct)
		}
	}
	if !matricesClose(t, correct, [][]float64{{0, 3}}) {
		t.Errorf("expected about [[0, 3]], got %v", correct)
	}
}

// TestMLPFixtureReadsItsInput guards the second half. withGeluBump left
// mlp.c_fc.weight zero, so the MLP ignored the row and skipping ln_2 changed
// nothing.
func TestMLPFixtureReadsItsInput(t *testing.T) {
	input := [][]float64{{1, 2}}
	residual := blockOutput(t, residualLayer(0, 2), 0, input)
	withMLP := blockOutput(t, withMLPBump(0, 2), 0, input)
	if matricesClose(t, residual, withMLP) {
		t.Fatalf("the MLP changes nothing: %v", withMLP)
	}
	// About [0, 3] into ln_2 gives [-1, 1], and GELU of those is
	// -0.1588 and 0.8412, added back onto [0, 3].
	if !matricesClose(t, withMLP, [][]float64{{-0.15878819, 3.84116958}}) {
		t.Fatalf("expected about [[-0.1588, 3.8412]], got %v", withMLP)
	}
	if !slices.Contains([]int{2}, len(withMLP[0])) {
		t.Fatalf("expected rows of 2 numbers, got %d", len(withMLP[0]))
	}
}
