// Package gpt2text turns GPT-2 token ids back into text, so the tester can
// show learners what their engine wrote.
//
// GPT-2's vocab.json maps each token to a string in which every byte is
// replaced by a printable character (a space becomes "Ġ"). Decoding looks up
// each id's string, maps the characters back to bytes, and joins the bytes.
package gpt2text

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
)

//go:embed vocab.json
var vocabJSON []byte

var (
	loadOnce  sync.Once
	tokens    map[int]string
	charBytes map[rune]byte
	loadErr   error
)

func load() {
	var vocab map[string]int
	if err := json.Unmarshal(vocabJSON, &vocab); err != nil {
		loadErr = err
		return
	}
	tokens = make(map[int]string, len(vocab))
	for text, id := range vocab {
		tokens[id] = text
	}
	charBytes = byteDecoder()
}

// byteDecoder inverts GPT-2's bytes_to_unicode table. Printable bytes map to
// themselves, and the rest map to code points from 256 upward, in byte order.
func byteDecoder() map[rune]byte {
	out := make(map[rune]byte, 256)
	printable := func(b int) bool {
		return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
	}
	next := 0
	for b := 0; b < 256; b++ {
		if printable(b) {
			out[rune(b)] = byte(b)
			continue
		}
		out[rune(256+next)] = byte(b)
		next++
	}
	return out
}

// Decode returns the text for ids. An id outside the vocabulary is an error.
func Decode(ids []int) (string, error) {
	loadOnce.Do(load)
	if loadErr != nil {
		return "", loadErr
	}
	var out []byte
	for _, id := range ids {
		text, ok := tokens[id]
		if !ok {
			return "", fmt.Errorf("token id %d is not in GPT-2's vocabulary", id)
		}
		for _, char := range text {
			b, ok := charBytes[char]
			if !ok {
				return "", fmt.Errorf("token id %d has an unexpected character %q", id, char)
			}
			out = append(out, b)
		}
	}
	return strings.ToValidUTF8(string(out), "�"), nil
}
