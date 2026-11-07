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
