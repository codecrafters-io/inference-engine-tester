package internal

import (
	"github.com/codecrafters-io/inference-engine-tester/internal/safetensors"
	"github.com/codecrafters-io/tester-utils/test_case_harness"
)

func testReadSafetensors(harness *test_case_harness.TestCaseHarness) error {
	cases := []inspectCase{
		// The stage description's worked example.
		{
			name: "scale",
			hint: "Print a JSON array with one entry per tensor. Each entry has the tensor's name, dtype, shape, and first, its first numbers.",
			tensors: []safetensors.Tensor{{
				Name:  "scale",
				DType: "F32",
				Shape: []int{4},
				Data:  safetensors.Float32Data(1, 0.5, 0.25, 0.125),
			}},
		},
		// More than four numbers, in two rows. The other cases have four or
		// fewer, so they can't tell "the first 4" from "all of them".
		{
			name: "first-four",
			hint: "first holds only the tensor's first 4 numbers, in the order the loader returns them. shape is printed as the file gives it, like [2, 3].",
			tensors: []safetensors.Tensor{{
				Name:  "table",
				DType: "F32",
				Shape: []int{2, 3},
				Data:  safetensors.Float32Data(1, 2, 3, 4, 5, 6),
			}},
		},
		{
			name: "name-order",
			hint: "Sort the tensors by name before printing them.",
			tensors: []safetensors.Tensor{
				{Name: "z", DType: "F32", Shape: []int{1}, Data: safetensors.Float32Data(3)},
				{Name: "a", DType: "F32", Shape: []int{1}, Data: safetensors.Float32Data(1)},
			},
		},
		{
			name: "data-offset",
			hint: "Each tensor has its own numbers. Print the values the loader returned under that tensor's name.",
			tensors: []safetensors.Tensor{
				{Name: "first", DType: "F32", Shape: []int{2}, Data: safetensors.Float32Data(9, 8)},
				{Name: "second", DType: "F32", Shape: []int{2}, Data: safetensors.Float32Data(1, 2)},
			},
		},
		{
			name: "integer",
			hint: "An I32 tensor holds whole numbers. Print its dtype as I32 and its values as the loader returns them.",
			tensors: []safetensors.Tensor{{
				Name:  "ids",
				DType: "I32",
				Shape: []int{3},
				Data:  safetensors.Int32Data(1, 2, 3),
			}},
		},
	}
	for _, testCase := range cases {
		if err := runInspect(harness, testCase); err != nil {
			return err
		}
	}
	return nil
}
