package gpt2text

import "testing"

// The ids were found by looking each token's string up in GPT-2's vocab.json,
// independently of the byte table this package inverts.
func TestDecodeMatchesGPT2Tokenizer(t *testing.T) {
	cases := []struct {
		ids  []int
		want string
	}{
		{[]int{464, 3139, 286, 4881, 318}, "The capital of France is"},
		{[]int{262, 3139, 286, 262, 4141, 2066}, " the capital of the French Republic"},
		{[]int{198, 198}, "\n\n"},
		{[]int{40, 1101}, "I'm"},
	}
	for _, c := range cases {
		got, err := Decode(c.ids)
		if err != nil {
			t.Fatal(err)
		}
		if got != c.want {
			t.Fatalf("Decode(%v) = %q, want %q", c.ids, got, c.want)
		}
	}
}

func TestDecodeRejectsUnknownID(t *testing.T) {
	if _, err := Decode([]int{50257}); err == nil {
		t.Fatal("expected an error for id 50257")
	}
}
