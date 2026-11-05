package internal

import (
	"github.com/codecrafters-io/inference-engine-tester/internal/safetensors"
	"github.com/codecrafters-io/tester-utils/test_case_harness"
)

func testReadSafetensors(harness *test_case_harness.TestCaseHarness) error {
	cases := []inspectCase{
		{
			name: "scale",
			hint: "Read the 8-byte little-endian header length first, then parse that many bytes as JSON.",
			tensors: []safetensors.Tensor{{
				Name:  "scale",
				DType: "F32",
				Shape: []int{4},
				Data:  safetensors.Float32Data(1, 0.5, 0.25, 0.125),
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
			hint: "Read each tensor at its own data_offsets, counted from the first byte after the header.",
			tensors: []safetensors.Tensor{
				{Name: "first", DType: "F32", Shape: []int{2}, Data: safetensors.Float32Data(9, 8)},
				{Name: "second", DType: "F32", Shape: []int{2}, Data: safetensors.Float32Data(1, 2)},
			},
		},
		{
			name: "integer",
			hint: "Decode I32 values as little-endian 32-bit integers, not as floats.",
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
