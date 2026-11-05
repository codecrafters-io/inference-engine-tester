package internal

import (
	"path/filepath"
	"testing"
)

func TestReadSafetensorsReference(t *testing.T) {
	repo, err := filepath.Abs("test_helpers/pass_all")
	if err != nil {
		t.Fatal(err)
	}
	code := RunCLI(map[string]string{
		"CODECRAFTERS_REPOSITORY_DIR":  repo,
		"CODECRAFTERS_TEST_CASES_JSON": `[{"slug":"zf8","tester_log_prefix":"stage-1","title":"Read safetensors"},{"slug":"pq6","tester_log_prefix":"stage-2","title":"Embed tokens"},{"slug":"mo0","tester_log_prefix":"stage-3","title":"Causal attention"},{"slug":"hu4","tester_log_prefix":"stage-4","title":"Transformer block"},{"slug":"wi3","tester_log_prefix":"stage-5","title":"Logits"},{"slug":"hh8","tester_log_prefix":"stage-6","title":"Greedy decode"},{"slug":"df2","tester_log_prefix":"stage-7","title":"Sample a token"},{"slug":"rq5","tester_log_prefix":"stage-8","title":"KV cache"}]`,
	})
	if code != 0 {
		t.Fatalf("reference solution failed, tester exit code %d", code)
	}
}
