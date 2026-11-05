# Inference Engine Challenge Tester

This is a program that validates your progress on the Inference Engine challenge.

# Requirements for binary

- Following environment variables:
  - `CODECRAFTERS_REPOSITORY_DIR` - root of the user's code submission
  - `CODECRAFTERS_TEST_CASES_JSON` - test cases in JSON format
  - `CODECRAFTERS_GPT2_WEIGHTS_PATH` - optional, the path of GPT-2's
    `model.safetensors` for the "Run GPT-2" stage, for developing the tester.

# GPT-2's weights in the test environment

Only the "Run GPT-2" stage (`zr7`) needs GPT-2's 548 MB weights. When it starts,
the tester looks for them at `$TMPDIR/codecrafters-gpt2/model.safetensors` on
the container's disk. If they're missing, it downloads them from
`gpt2WeightsURL` in `internal/gpt2_weights.go`, streaming to disk rather than
memory, and keeps the file only if its size and SHA-256 match. A later run in
the same pod reuses the file, and a new pod downloads it again. The download
times out after 3 minutes, well inside the pod's 10-minute limit.

The learner's program runs in the same pod, so it shares the pod's 2560Mi
memory limit. Stage 9's description tells learners about the limit, and the
tester adds a memory hint when the program dies without printing anything.

# User code requirements

- A binary named `your_program.sh` that executes the inference engine command.
- A file named `codecrafters.yml`, with the following values:
  - `debug`

# Testing the tester

`make test` runs the unit tests and every stage except "Run GPT-2" against the
reference solution in `internal/test_helpers/pass_all`. It needs no network.

## The "Run GPT-2" stage

This stage (`zr7`) runs the learner's engine on GPT-2's real 548 MB weights, so
it has its own script:

```bash
make test_gpt2
```

The script `scripts/test_gpt2_stage.sh`:

1. Downloads GPT-2's weights from Hugging Face into
   `~/.cache/codecrafters/gpt2/`, once, and checks the file's SHA-256. Set
   `GPT2_WEIGHTS_DIR` to keep the file somewhere else.
2. Checks that the Go oracle continues "The capital of France is" with GPT-2's
   six ids, " the capital of the French Republic".
3. Builds the tester and runs stage `zr7` against the reference solution. The
   Python reference takes about 45 seconds.

To run the stage against another solution, pass its directory:

```bash
scripts/test_gpt2_stage.sh path/to/solution
```
