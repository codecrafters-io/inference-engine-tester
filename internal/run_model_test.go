package internal

import (
	"strings"
	"testing"
)

func TestParseJSONShowsANonJSONFirstLine(t *testing.T) {
	stdout := "['/app/app/main.py', 'embed']\n[[13, 24], [31, 42]]\n"
	_, err := parseJSON[[][]float64]([]byte(stdout))
	if err == nil {
		t.Fatal("expected an error for a Python list on stdout")
	}
	for _, want := range []string{"['/app/app/main.py', 'embed']", "send debug output to stderr", "double quotes"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q doesn't mention %q", err, want)
		}
	}
}

func TestParseJSONSkipsTheQuoteHintForOtherText(t *testing.T) {
	_, err := parseJSON[[][]float64]([]byte("embedding done\n"))
	if err == nil {
		t.Fatal("expected an error for plain text on stdout")
	}
	if strings.Contains(err.Error(), "double quotes") {
		t.Errorf("error %q gives the quote hint for text that isn't a list", err)
	}
}

func TestParseJSONReportsTheWrongShapeAsJSON(t *testing.T) {
	cases := []struct {
		stdout string
		parse  func([]byte) error
		want   string
	}{
		{"[13, 24]", func(b []byte) error { _, err := parseJSON[[][]float64](b); return err },
			"expected an array of arrays of numbers on stdout, but your program's JSON has a number where the tester expects an array of numbers"},
		{"[464, 1.5]", func(b []byte) error { _, err := parseJSON[[]int](b); return err },
			"has a number 1.5 where the tester expects a whole number"},
		{`{"logits": [1]}`, func(b []byte) error { _, err := parseJSON[[]float64](b); return err },
			"has an object where the tester expects an array of numbers"},
	}
	for _, c := range cases {
		err := c.parse([]byte(c.stdout))
		if err == nil {
			t.Fatalf("expected an error for %s", c.stdout)
		}
		if !strings.Contains(err.Error(), c.want) {
			t.Errorf("error for %s is %q, want it to contain %q", c.stdout, err, c.want)
		}
		if strings.Contains(err.Error(), "isn't JSON") || strings.Contains(err.Error(), "stderr") {
			t.Errorf("error for %s calls valid JSON invalid: %q", c.stdout, err)
		}
	}
}

// TestDiffsShowBothValues checks that every comparison a learner can hit
// prints the whole expected and printed values, not only the first element
// that differs. One element out of context looks the same whether the result
// is transposed, shifted by one, or simply wrong.
func TestDiffsShowBothValues(t *testing.T) {
	cases := []struct {
		name     string
		message  string
		contains []string
	}{
		{
			name:     "matrix values",
			message:  diffMatrix([][]float64{{1, 2}}, [][]float64{{1, 3}}, 1e-6, 0),
			contains: []string{"row 0, column 1", "expected: [[1,2]]", "got: [[1,3]]"},
		},
		{
			name:     "matrix rows",
			message:  diffMatrix([][]float64{{1}, {2}}, [][]float64{{1}}, 1e-6, 0),
			contains: []string{"expected 2 rows, got 1", "expected: [[1],[2]]", "got: [[1]]"},
		},
		{
			name:     "matrix transposed",
			message:  diffMatrix([][]float64{{1, 2}}, [][]float64{{1}, {2}}, 1e-6, 0),
			contains: []string{"expected: [[1,2]]", "got: [[1],[2]]"},
		},
		{
			name:     "vector",
			message:  diffVector([]float64{1, 2}, []float64{1, 5}, 1e-6, 0),
			contains: []string{"value 1", "expected: [1,2]", "got: [1,5]"},
		},
		{
			name:     "ids",
			message:  diffInts([]int{1, 2, 3}, []int{1, 3, 3}),
			contains: []string{"token 1", "expected: [1,2,3]", "got: [1,3,3]"},
		},
	}
	for _, testCase := range cases {
		for _, want := range testCase.contains {
			if !strings.Contains(testCase.message, want) {
				t.Errorf("%s: message is missing %q:\n%s", testCase.name, want, testCase.message)
			}
		}
	}
}

// TestDiffsStaySilentWhenEqual guards against the formatting above reporting a
// difference that isn't there.
func TestDiffsStaySilentWhenEqual(t *testing.T) {
	if message := diffMatrix([][]float64{{1, 2}}, [][]float64{{1, 2}}, 1e-6, 0); message != "" {
		t.Errorf("equal matrices reported %q", message)
	}
	if message := diffVector([]float64{1}, []float64{1}, 1e-6, 0); message != "" {
		t.Errorf("equal vectors reported %q", message)
	}
	if message := diffInts([]int{1}, []int{1}); message != "" {
		t.Errorf("equal ids reported %q", message)
	}
}

// TestLargeValuesAreLeftOut keeps a difference from being buried. A pair of
// clipped values is worse than the precise difference alone, because the
// element that differs is usually past the clip.
func TestLargeValuesAreLeftOut(t *testing.T) {
	big := make([]int, 500)
	other := make([]int, 500)
	other[499] = 1
	message := diffInts(big, other)
	if !strings.Contains(message, "token 499") {
		t.Errorf("message should still say where it differs:\n%s", message)
	}
	if strings.Contains(message, "expected: [") {
		t.Errorf("a 500-value result should not be printed:\n%s", message)
	}
	if strings.Contains(message, "...") {
		t.Errorf("values should be left out, not clipped:\n%s", message)
	}
}
