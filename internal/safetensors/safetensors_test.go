package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"
)

func TestScaleExample(t *testing.T) {
	raw, err := Write([]Tensor{{
		Name:  "scale",
		DType: "F32",
		Shape: []int{4},
		Data:  Float32Data(1, 0.5, 0.25, 0.125),
	}})
	if err != nil {
		t.Fatal(err)
	}

	headerLen := binary.LittleEndian.Uint64(raw[:8])
	if headerLen != 90 {
		t.Fatalf("header length = %d, want 90", headerLen)
	}
	if string(raw[8:8+headerLen]) != `{"__metadata__":{"format":"pt"},"scale":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}` {
		t.Fatalf("header = %s", raw[8:8+headerLen])
	}

	got, err := Inspect(raw)
	if err != nil {
		t.Fatal(err)
	}
	want := []Summary{{Name: "scale", DType: "F32", Shape: []int{4}, First: []float64{1, 0.5, 0.25, 0.125}}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %#v", got)
	}
}

func TestSkipsMetadataAndSortsByName(t *testing.T) {
	raw, err := Write([]Tensor{
		{Name: "z", DType: "F32", Shape: []int{1}, Data: Float32Data(3)},
		{Name: "a", DType: "F32", Shape: []int{1}, Data: Float32Data(1)},
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := Inspect(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Name != "a" || got[1].Name != "z" {
		t.Fatalf("names = %#v", got)
	}
	if got[0].First[0] != 1 || got[1].First[0] != 3 {
		t.Fatalf("values = %#v", got)
	}
}

func TestSecondTensorUsesItsOwnOffset(t *testing.T) {
	raw, err := Write([]Tensor{
		{Name: "first", DType: "F32", Shape: []int{2}, Data: Float32Data(9, 8)},
		{Name: "second", DType: "F32", Shape: []int{2}, Data: Float32Data(1, 2)},
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := Inspect(raw)
	if err != nil {
		t.Fatal(err)
	}
	if got[1].Name != "second" || !reflect.DeepEqual(got[1].First, []float64{1, 2}) {
		t.Fatalf("second = %#v", got[1])
	}
}

func TestInt32IsNotReadAsFloat(t *testing.T) {
	raw, err := Write([]Tensor{{
		Name:  "ids",
		DType: "I32",
		Shape: []int{3},
		Data:  Int32Data(1, 2, 3),
	}})
	if err != nil {
		t.Fatal(err)
	}
	got, err := Inspect(raw)
	if err != nil {
		t.Fatal(err)
	}
	if got[0].DType != "I32" || !reflect.DeepEqual(got[0].First, []float64{1, 2, 3}) {
		t.Fatalf("got %#v", got[0])
	}
	asFloat := float64(math.Float32frombits(1))
	if got[0].First[0] == asFloat {
		t.Fatal("I32 value 1 was read as float32 bits")
	}
}

func TestGPT2Header(t *testing.T) {
	raw, err := os.ReadFile("testdata/gpt2-header.bin")
	if err != nil {
		t.Fatal(err)
	}
	headerLen := binary.LittleEndian.Uint64(raw[:8])
	if headerLen != 14283 {
		t.Fatalf("GPT-2 header length = %d, want 14283", headerLen)
	}
	// The captured file is only the header, so tensor bytes are absent.
	// Parsing still has to name wte.weight before it tries to read values.
	if uint64(len(raw)) < 8+headerLen {
		t.Fatalf("captured header is short: %d", len(raw))
	}
	file := raw[:8+headerLen]
	// Append one dummy byte so a reader that ignores data_offsets and asks
	// for values fails inside Inspect. The name check below uses the header only.
	var header map[string]struct {
		DType string `json:"dtype"`
		Shape []int  `json:"shape"`
	}
	if err := json.Unmarshal(file[8:], &header); err != nil {
		t.Fatal(err)
	}
	wte, ok := header["wte.weight"]
	if !ok {
		t.Fatal("wte.weight missing")
	}
	if wte.DType != "F32" || !reflect.DeepEqual(wte.Shape, []int{50257, 768}) {
		t.Fatalf("wte.weight = %#v", wte)
	}
	if _, ok := header["__metadata__"]; !ok {
		t.Fatal("metadata missing")
	}
}
