# Inference Engine Challenge Tester

This is a program that validates your progress on the Inference Engine challenge.

# Requirements for binary

- Following environment variables:
  - `CODECRAFTERS_REPOSITORY_DIR` - root of the user's code submission
  - `CODECRAFTERS_TEST_CASES_JSON` - test cases in JSON format
  - `CODECRAFTERS_GPT2_WEIGHTS_PATH` - optional, the path of GPT-2's
    `model.safetensors`, used by the "Run GPT-2" stage. Without it, the tester
    downloads the file once into the user cache directory and checks its SHA-256.

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
