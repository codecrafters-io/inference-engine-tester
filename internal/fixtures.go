package internal

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"slices"
	"strconv"
	"time"

	"github.com/codecrafters-io/inference-engine-tester/internal/gpt2"
)

// Each stage runs its worked example first, on the small fixed file the stage
// description walks through, and then cases on random models.
//
// The fixed files are there so a learner can check the description's numbers
// by hand. They can't be the whole test: their weights are mostly 0 and 1, so
// an engine that skips a multiply, a bias, or a layer norm's weight gets the
// same answer. A random model has no such blind spots. Before the tester uses
// one, it runs the oracle with each bug the stage should catch (gpt2.Bug) and
// draws again unless every one of them changes the answer by a wide margin.

// fixtureTries is how many random models a case may draw before giving up.
// The cases in this tester settle within a few dozen draws.
const fixtureTries = 20000

// newFixtureRand seeds from CODECRAFTERS_RANDOM_SEED when it is set, so a run
// can be repeated.
func newFixtureRand() *rand.Rand {
	if seed, err := strconv.ParseInt(os.Getenv("CODECRAFTERS_RANDOM_SEED"), 10, 64); err == nil {
		return rand.New(rand.NewSource(seed))
	}
	return rand.New(rand.NewSource(time.Now().UnixNano()))
}

// randomValues draws n numbers between -1 and 1 with two decimals, short
// enough to read in the tester's log.
func randomValues(rng *rand.Rand, n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = float64(rng.Intn(201)-100) / 100
	}
	return out
}

func randomRows(rng *rand.Rand, rows, cols int) [][]float64 {
	out := make([][]float64, rows)
	for i := range out {
		out[i] = randomValues(rng, cols)
	}
	return out
}

func randomIDs(rng *rand.Rand, n, vocab int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = rng.Intn(vocab)
	}
	return out
}

func randomTensor(rng *rand.Rand, name string, shape ...int) namedTensor {
	count := 1
	for _, dim := range shape {
		count *= dim
	}
	return namedTensor{name, shape, randomValues(rng, count)}
}

func randomAttention(rng *rand.Rand, layer, nEmbd int) []namedTensor {
	prefix := "h." + itoa(layer) + ".attn"
	return []namedTensor{
		randomTensor(rng, prefix+".c_attn.weight", nEmbd, 3*nEmbd),
		randomTensor(rng, prefix+".c_attn.bias", 3*nEmbd),
		randomTensor(rng, prefix+".c_proj.weight", nEmbd, nEmbd),
		randomTensor(rng, prefix+".c_proj.bias", nEmbd),
	}
}

// randomLayer is one block with every tensor random.
func randomLayer(rng *rand.Rand, layer, nEmbd int) []namedTensor {
	prefix := "h." + itoa(layer)
	parts := []namedTensor{
		randomTensor(rng, prefix+".ln_1.weight", nEmbd),
		randomTensor(rng, prefix+".ln_1.bias", nEmbd),
	}
	parts = append(parts, randomAttention(rng, layer, nEmbd)...)
	return append(parts,
		randomTensor(rng, prefix+".ln_2.weight", nEmbd),
		randomTensor(rng, prefix+".ln_2.bias", nEmbd),
		randomTensor(rng, prefix+".mlp.c_fc.weight", nEmbd, 4*nEmbd),
		randomTensor(rng, prefix+".mlp.c_fc.bias", 4*nEmbd),
		randomTensor(rng, prefix+".mlp.c_proj.weight", 4*nEmbd, nEmbd),
		randomTensor(rng, prefix+".mlp.c_proj.bias", nEmbd),
	)
}

// randomResidualLayer is randomLayer with ln_2 and the MLP zeroed. The
// "Residual connections" stage comes before the MLP, and a block whose MLP
// tensors are zero gives the same output with or without one.
func randomResidualLayer(rng *rand.Rand, layer, nEmbd int) []namedTensor {
	parts := randomLayer(rng, layer, nEmbd)
	prefix := "h." + itoa(layer)
	for i := range parts {
		switch parts[i].name {
		case prefix + ".ln_2.weight", prefix + ".ln_2.bias",
			prefix + ".mlp.c_fc.weight", prefix + ".mlp.c_fc.bias",
			prefix + ".mlp.c_proj.weight", prefix + ".mlp.c_proj.bias":
			parts[i].values = zeros(len(parts[i].values))
		}
	}
	return parts
}

// randomGPT is a whole model with every tensor random.
func randomGPT(rng *rand.Rand, cfg gpt2.Config) []namedTensor {
	parts := []namedTensor{
		randomTensor(rng, "wte.weight", cfg.VocabSize, cfg.NEmbd),
		randomTensor(rng, "wpe.weight", cfg.NPositions, cfg.NEmbd),
		randomTensor(rng, "ln_f.weight", cfg.NEmbd),
		randomTensor(rng, "ln_f.bias", cfg.NEmbd),
	}
	for layer := 0; layer < cfg.NLayer; layer++ {
		parts = append(parts, randomLayer(rng, layer, cfg.NEmbd)...)
	}
	return parts
}

func loadParts(parts []namedTensor) ([]byte, *gpt2.Weights) {
	file := mustModel(tensorsFrom(parts))
	weights, err := gpt2.Load(file)
	if err != nil {
		panic(err)
	}
	return file, weights
}

// settle draws until build accepts a draw. build returns false for a draw the
// tester can't use: one where a bug leaves the answer unchanged, or where two
// scores are too close to call.
func settle[T any](rng *rand.Rand, name string, build func(*rand.Rand) (T, bool)) T {
	for range fixtureTries {
		if value, ok := build(rng); ok {
			return value
		}
	}
	panic(fmt.Sprintf("%s: no usable random model in %d draws", name, fixtureTries))
}

// mutant is what the oracle prints with one bug switched on.
type mutant[T any] struct {
	bug    gpt2.Bug
	output T
}

// mutantsOf runs the oracle once for each bug. A bug that makes the oracle
// fail on this input, like a position past the end of wpe.weight, is left out.
func mutantsOf[T any](weights *gpt2.Weights, bugs []gpt2.Bug, run func(*gpt2.Weights) (T, error)) []mutant[T] {
	out := make([]mutant[T], 0, len(bugs))
	for _, bug := range bugs {
		output, err := run(weights.WithBug(bug))
		if err != nil {
			continue
		}
		out = append(out, mutant[T]{bug, output})
	}
	return out
}

// allCaught reports whether every bug ran and gave an answer the comparison
// tells apart from the expected one.
func allCaught[T any](bugs []gpt2.Bug, mutants []mutant[T], expected T, apart func(a, b T) bool) bool {
	if len(mutants) != len(bugs) {
		return false
	}
	for _, m := range mutants {
		if !apart(expected, m.output) {
			return false
		}
	}
	return true
}

// tellApart reports whether every bug's output differs from every other
// bug's. The generate stages print a handful of token ids, and two different
// bugs can easily print the same ones. A case whose bugs all print different
// ids can say which bug it is looking at.
func tellApart[T any](mutants []mutant[T], apart func(a, b T) bool) bool {
	for i := range mutants {
		for j := range i {
			if !apart(mutants[i].output, mutants[j].output) {
				return false
			}
		}
	}
	return true
}

// diagnose picks the hint for a wrong answer: the hint of the bug whose output
// the learner's matches, or the case's own hint when no bug's does. When bugs
// with different hints all match, naming one of them would be a guess, so the
// case's own hint is used then too.
func diagnose[T any](got T, mutants []mutant[T], same func(a, b T) bool, fallback string) string {
	hint := ""
	for _, m := range mutants {
		if !same(m.output, got) {
			continue
		}
		if hint != "" && hint != bugHints[m.bug] {
			return fallback
		}
		hint = bugHints[m.bug]
	}
	if hint == "" {
		return fallback
	}
	return hint
}

// apartBy is the margin a bug has to move at least one number by. It is far
// above the tolerances the stages compare with, so no float type or order of
// operations can blur a bug into a pass.
const apartBy = 0.02

func matricesApart(a, b [][]float64) bool {
	if len(a) != len(b) {
		return true
	}
	for i := range a {
		if vectorsApart(a[i], b[i]) {
			return true
		}
	}
	return false
}

func vectorsApart(a, b []float64) bool {
	if len(a) != len(b) {
		return true
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > apartBy*math.Max(1, math.Abs(a[i])) {
			return true
		}
	}
	return false
}

func idsApart(a, b []int) bool {
	return !slices.Equal(a, b)
}

func sameMatrix(atol, rtol float64) func(a, b [][]float64) bool {
	return func(a, b [][]float64) bool { return diffMatrix(a, b, atol, rtol) == "" }
}

func sameVector(atol, rtol float64) func(a, b []float64) bool {
	return func(a, b []float64) bool { return diffVector(a, b, atol, rtol) == "" }
}
