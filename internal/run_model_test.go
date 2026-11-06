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
