package internal

import (
	"math"
	"math/rand"
	"path/filepath"
	"slices"
	"strconv"
	"testing"

	"github.com/codecrafters-io/inference-engine-tester/internal/gpt2"
)

// fixtureSeeds is how many different random runs the tests below build.
const fixtureSeeds = 200

// caughtBy lists, for one seed, the bugs each stage's cases tell apart from a
// correct answer. Building the cases also proves they settle: a case that
// can't find a usable random model panics.
func caughtBy(seed int64) map[string][]gpt2.Bug {
	rng := rand.New(rand.NewSource(seed))
	out := map[string][]gpt2.Bug{}
	note := func(stage string, bug gpt2.Bug) {
		if !slices.Contains(out[stage], bug) {
			out[stage] = append(out[stage], bug)
		}
	}
	for _, c := range embedCases(rng) {
		_, w := loadParts(c.parts)
		expected, _ := gpt2.Embed(w, c.ids)
		for _, m := range c.mutants(w) {
			if matricesApart(expected, m.output) {
				note("pq6", m.bug)
			}
		}
	}
	matrixStages := []struct {
		stage  string
		oracle matrixOracle
		cases  []matrixCase
	}{
		{"mo0", gpt2.Attention, attentionCases(rng)},
		{"bx7", gpt2.Attention, headsCases(rng)},
		{"vb8", gpt2.Block, residualCases(rng)},
		{"hu4", gpt2.Block, transformerBlockCases(rng)},
	}
	for _, s := range matrixStages {
		for _, c := range s.cases {
			_, w := loadParts(c.parts)
			expected, _ := c.run(s.oracle, w)
			for _, m := range c.mutants(s.oracle, w) {
				if matricesApart(expected, m.output) {
					note(s.stage, m.bug)
				}
			}
		}
	}
	for _, c := range logitsCases(rng) {
		_, w := loadParts(c.parts)
		expected, _ := gpt2.Logits(w, c.cfg, c.ids)
		for _, m := range c.mutants(w) {
			if vectorsApart(expected, m.output) {
				note("wi3", m.bug)
			}
		}
	}
	generateStages := map[string][]generateCase{
		"hh8": greedyCases(rng),
		"df2": samplingCases(rng),
		"rq5": kvCacheCases(rng),
	}
	for stage, cases := range generateStages {
		for _, c := range cases {
			_, w := loadParts(c.parts)
			expected, _ := c.tokens(w)
			for _, m := range c.mutants(w) {
				if idsApart(expected, m.output) {
					note(stage, m.bug)
				}
			}
		}
	}
	return out
}

// stageBugs is the contract: the mistakes each stage has to catch on every
// run, whatever the random numbers. A stage introduces the steps these bugs
// get wrong, so a learner who makes one should hear about it there, not nine
// stages later on GPT-2's real weights.
var stageBugs = map[string][]gpt2.Bug{
	"pq6": embedBugs,
	"mo0": bugList(attentionBugs, []gpt2.Bug{gpt2.LayerZero}),
	"bx7": bugList(attentionBugs, headBugs, []gpt2.Bug{gpt2.LayerZero}),
	"vb8": bugList(residualBugs, []gpt2.Bug{gpt2.LayerZero}),
	"hu4": bugList(residualBugs, mlpBugs, []gpt2.Bug{gpt2.LayerZero}),
	"wi3": bugList(logitsBugs, []gpt2.Bug{gpt2.LogitsOneLayer}),
	"hh8": {gpt2.PickTieHigher, gpt2.GenNoFeedback},
	"df2": bugList(topPBugs, temperatureBugs),
	"rq5": cacheBugs,
}

func TestEveryStageCatchesItsBugsOnEverySeed(t *testing.T) {
	for seed := int64(1); seed <= fixtureSeeds; seed++ {
		caught := caughtBy(seed)
		for stage, bugs := range stageBugs {
			for _, bug := range bugs {
				if !slices.Contains(caught[stage], bug) {
					t.Errorf("seed %d: stage %s lets bug %d through (%s)", seed, stage, bug, bugHints[bug])
				}
			}
		}
	}
}

// TestGenerateHintsAreNotGuesses checks that a random generate case never has
// two bugs printing the same ids, since the ids are all the tester has to go
// on when it names the bug.
func TestGenerateHintsAreNotGuesses(t *testing.T) {
	for seed := int64(1); seed <= fixtureSeeds; seed++ {
		rng := rand.New(rand.NewSource(seed))
		greedy, sampling, cache := greedyCases(rng), samplingCases(rng), kvCacheCases(rng)
		random := []generateCase{greedy[len(greedy)-1], sampling[1], sampling[2], cache[len(cache)-1]}
		for _, c := range random {
			_, w := loadParts(c.parts)
			for _, m := range c.mutants(w) {
				if hint := diagnose(m.output, c.mutants(w), slices.Equal[[]int], c.hint); hint != bugHints[m.bug] {
					t.Errorf("seed %d, %s: bug %d got the hint %q", seed, c.name, m.bug, hint)
				}
			}
		}
	}
}

func TestEveryBugHasAStageAndAHint(t *testing.T) {
	for _, bug := range gpt2.AllBugs() {
		if bugHints[bug] == "" {
			t.Errorf("bug %d has no hint", bug)
		}
		found := false
		for _, bugs := range stageBugs {
			found = found || slices.Contains(bugs, bug)
		}
		if !found {
			t.Errorf("bug %d (%s) isn't caught by any stage", bug, bugHints[bug])
		}
	}
}

// TestOneHeadEngineFailsTheHeadsStage is the check that the heads stage asks
// for something: an engine that passed the stage before, with one head across
// the whole row, has to fail every case here.
func TestOneHeadEngineFailsTheHeadsStage(t *testing.T) {
	for seed := int64(1); seed <= fixtureSeeds; seed++ {
		for _, c := range headsCases(rand.New(rand.NewSource(seed))) {
			_, w := loadParts(c.parts)
			expected, _ := c.run(gpt2.Attention, w)
			oneHead, _ := c.run(gpt2.Attention, w.WithBug(gpt2.AttnOneHead))
			if !matricesApart(expected, oneHead) {
				t.Errorf("seed %d, %s: a one-head engine prints the expected output", seed, c.name)
			}
		}
	}
}

// TestHeadsExampleMatchesTheDescription pins the worked example the heads
// stage's description prints.
func TestHeadsExampleMatchesTheDescription(t *testing.T) {
	c := headsCases(rand.New(rand.NewSource(1)))[0]
	_, w := loadParts(c.parts)
	got, _ := c.run(gpt2.Attention, w)
	want := [][]float64{{1, 0}, {0.5, 0.7310585786300049}}
	if diffMatrix(want, got, 1e-12, 0) != "" {
		t.Errorf("got %v, want %v", got, want)
	}
	oneHead, _ := c.run(gpt2.Attention, w.WithBug(gpt2.AttnOneHead))
	if math.Abs(oneHead[1][0]-0.330) > 0.0005 || math.Abs(oneHead[1][1]-0.670) > 0.0005 {
		t.Errorf("the description says one head gives [0.330, 0.670], got %v", oneHead[1])
	}
}

// TestDiagnoseNamesTheBug feeds each bug's own output back in as if a learner
// had printed it. The hint has to be that bug's, or the hint of another bug
// that prints exactly the same thing on this case.
func TestDiagnoseNamesTheBug(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	same := sameMatrix(1e-4, 0)
	for _, c := range slices.Concat(attentionCases(rng), headsCases(rng)) {
		_, w := loadParts(c.parts)
		mutants := c.mutants(gpt2.Attention, w)
		for _, m := range mutants {
			hint := diagnose(m.output, mutants, same, c.hint)
			twins := 0
			for _, other := range mutants {
				if same(other.output, m.output) {
					twins++
				}
			}
			// The worked example is too small to tell every bug apart, and
			// there the case's own hint is the honest answer.
			want := bugHints[m.bug]
			if twins > 1 {
				want = c.hint
			}
			if hint != want {
				t.Errorf("%s: bug %d got the hint %q, want %q", c.name, m.bug, hint, want)
			}
		}
		expected, _ := c.run(gpt2.Attention, w)
		off := [][]float64{slices.Clone(expected[0])}
		off[0][0] += 0.5
		if hint := diagnose(off, mutants, same, c.hint); hint != c.hint {
			t.Errorf("%s: an output no bug explains got the hint %q, not the case's own", c.name, hint)
		}
	}
}

// TestRandomCasesAreNotTheOldBlindSpots pins the properties the fixed files
// lacked: rows wide enough to keep their shape through a layer norm, a second
// layer in the KV cache stage, and more than one sampled token.
func TestRandomCasesAreNotTheOldBlindSpots(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, c := range slices.Concat(residualCases(rng)[1:], transformerBlockCases(rng)[1:]) {
		if c.cfg.NEmbd < 4 {
			t.Errorf("block case %s has rows of %d numbers", c.name, c.cfg.NEmbd)
		}
	}
	cache := kvCacheCases(rng)
	if last := cache[len(cache)-1]; last.cfg.NLayer < 2 {
		t.Errorf("the KV cache stage's random case has %d layer", last.cfg.NLayer)
	}
	for _, c := range samplingCases(rng)[1:] {
		if c.count < 2 {
			t.Errorf("sampling case %s picks %d token", c.name, c.count)
		}
	}
	greedy := greedyCases(rng)
	_, w := loadParts(greedy[len(greedy)-1].parts)
	ids, _ := greedy[len(greedy)-1].tokens(w)
	if slices.Min(ids) == slices.Max(ids) {
		t.Errorf("the greedy stage's random case picks the same token every time: %v", ids)
	}
}

// TestReferencePassesOnSeveralSeeds runs the reference solution against whole
// random runs, since a random case that a correct engine fails is worse than
// a bug the tester misses.
func TestReferencePassesOnSeveralSeeds(t *testing.T) {
	repo, err := filepath.Abs("test_helpers/pass_all")
	if err != nil {
		t.Fatal(err)
	}
	for seed := 1; seed <= 5; seed++ {
		t.Setenv("CODECRAFTERS_RANDOM_SEED", strconv.Itoa(seed))
		code := RunCLI(map[string]string{
			"CODECRAFTERS_REPOSITORY_DIR":  repo,
			"CODECRAFTERS_TEST_CASES_JSON": `[{"slug":"pq6","tester_log_prefix":"stage-2","title":"Embed tokens"},{"slug":"mo0","tester_log_prefix":"stage-3","title":"Causal attention"},{"slug":"bx7","tester_log_prefix":"stage-3b","title":"Multi-head attention"},{"slug":"vb8","tester_log_prefix":"stage-4","title":"Residual connections"},{"slug":"hu4","tester_log_prefix":"stage-5","title":"Transformer block"},{"slug":"wi3","tester_log_prefix":"stage-6","title":"Logits"},{"slug":"hh8","tester_log_prefix":"stage-7","title":"Greedy decode"},{"slug":"df2","tester_log_prefix":"stage-8","title":"Sample a token"},{"slug":"rq5","tester_log_prefix":"stage-9","title":"KV cache"}]`,
		})
		if code != 0 {
			t.Fatalf("seed %d: reference solution failed, tester exit code %d", seed, code)
		}
	}
}
