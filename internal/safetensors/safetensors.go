// Package safetensors reads and writes the file format used by GPT-2's weights.
package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"sort"
)

// Tensor is one named array in a safetensors file.
type Tensor struct {
	Name  string
	DType string
	Shape []int
	Data  []byte
}

// Summary is what the inspect command prints for one tensor.
type Summary struct {
	Name  string    `json:"name"`
	DType string    `json:"dtype"`
	Shape []int     `json:"shape"`
	First []float64 `json:"first"`
}

// Write encodes tensors in the given order. The header is not padded, matching
// GPT-2's file, whose header length is 14283.
func Write(tensors []Tensor) ([]byte, error) {
	var header bytes.Buffer
	header.WriteString(`{"__metadata__":{"format":"pt"}`)
	offset := 0
	var data bytes.Buffer
	for _, tensor := range tensors {
		size, err := dtypeSize(tensor.DType)
		if err != nil {
			return nil, err
		}
		count := numel(tensor.Shape)
		if len(tensor.Data) != count*size {
			return nil, fmt.Errorf("%s: want %d data bytes, got %d", tensor.Name, count*size, len(tensor.Data))
		}
		encoded, err := json.Marshal(struct {
			DType       string `json:"dtype"`
			Shape       []int  `json:"shape"`
			DataOffsets [2]int `json:"data_offsets"`
		}{tensor.DType, tensor.Shape, [2]int{offset, offset + len(tensor.Data)}})
		if err != nil {
			return nil, err
		}
		fmt.Fprintf(&header, ",%q:%s", tensor.Name, encoded)
		data.Write(tensor.Data)
		offset += len(tensor.Data)
	}
	header.WriteByte('}')

	var out bytes.Buffer
	if err := binary.Write(&out, binary.LittleEndian, uint64(header.Len())); err != nil {
		return nil, err
	}
	out.Write(header.Bytes())
	out.Write(data.Bytes())
	return out.Bytes(), nil
}

// Read returns every tensor in the file. __metadata__ is skipped.
func Read(file []byte) ([]Tensor, error) {
	header, data, err := parseFile(file)
	if err != nil {
		return nil, err
	}
	out := make([]Tensor, 0, len(header))
	for name, raw := range header {
		if name == "__metadata__" {
			continue
		}
		var info struct {
			DType       string `json:"dtype"`
			Shape       []int  `json:"shape"`
			DataOffsets [2]int `json:"data_offsets"`
		}
		if err := json.Unmarshal(raw, &info); err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		if info.DataOffsets[0] < 0 || info.DataOffsets[1] > len(data) || info.DataOffsets[0] > info.DataOffsets[1] {
			return nil, fmt.Errorf("%s: data_offsets %v are outside the data section", name, info.DataOffsets)
		}
		copied := append([]byte(nil), data[info.DataOffsets[0]:info.DataOffsets[1]]...)
		out = append(out, Tensor{Name: name, DType: info.DType, Shape: append([]int(nil), info.Shape...), Data: copied})
	}
	return out, nil
}

// Floats decodes an F32 tensor as float64 values in storage order.
func (t Tensor) Floats() ([]float64, error) {
	if t.DType != "F32" {
		return nil, fmt.Errorf("%s: want F32, got %s", t.Name, t.DType)
	}
	if len(t.Data)%4 != 0 {
		return nil, fmt.Errorf("%s: data length %d is not a multiple of 4", t.Name, len(t.Data))
	}
	out := make([]float64, len(t.Data)/4)
	for i := range out {
		bits := binary.LittleEndian.Uint32(t.Data[i*4 : (i+1)*4])
		out[i] = float64(math.Float32frombits(bits))
	}
	return out, nil
}

func parseFile(file []byte) (map[string]json.RawMessage, []byte, error) {
	if len(file) < 8 {
		return nil, nil, fmt.Errorf("file is shorter than the 8-byte header length")
	}
	headerLen := binary.LittleEndian.Uint64(file[:8])
	if uint64(len(file)) < 8+headerLen {
		return nil, nil, fmt.Errorf("header length %d runs past the end of the file", headerLen)
	}
	var header map[string]json.RawMessage
	if err := json.Unmarshal(file[8:8+headerLen], &header); err != nil {
		return nil, nil, fmt.Errorf("header is not JSON: %w", err)
	}
	return header, file[8+headerLen:], nil
}

// Inspect reads a safetensors file and returns one summary per tensor, sorted
// by name. __metadata__ is skipped. First holds at most 4 values.
func Inspect(file []byte) ([]Summary, error) {
	header, data, err := parseFile(file)
	if err != nil {
		return nil, err
	}

	names := make([]string, 0, len(header))
	for name := range header {
		if name == "__metadata__" {
			continue
		}
		names = append(names, name)
	}
	sort.Strings(names)

	out := make([]Summary, 0, len(names))
	for _, name := range names {
		var info struct {
			DType       string `json:"dtype"`
			Shape       []int  `json:"shape"`
			DataOffsets [2]int `json:"data_offsets"`
		}
		if err := json.Unmarshal(header[name], &info); err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		if info.DataOffsets[0] < 0 || info.DataOffsets[1] > len(data) || info.DataOffsets[0] > info.DataOffsets[1] {
			return nil, fmt.Errorf("%s: data_offsets %v are outside the data section", name, info.DataOffsets)
		}
		first, err := firstValues(info.DType, data[info.DataOffsets[0]:info.DataOffsets[1]], 4)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		out = append(out, Summary{Name: name, DType: info.DType, Shape: info.Shape, First: first})
	}
	return out, nil
}

func Float32Data(values ...float32) []byte {
	out := make([]byte, 4*len(values))
	for i, value := range values {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(value))
	}
	return out
}

func Int32Data(values ...int32) []byte {
	out := make([]byte, 4*len(values))
	for i, value := range values {
		binary.LittleEndian.PutUint32(out[i*4:], uint32(value))
	}
	return out
}

func firstValues(dtype string, data []byte, n int) ([]float64, error) {
	size, err := dtypeSize(dtype)
	if err != nil {
		return nil, err
	}
	if len(data)%size != 0 {
		return nil, fmt.Errorf("data length %d is not a multiple of %d", len(data), size)
	}
	count := len(data) / size
	if n > count {
		n = count
	}
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		chunk := data[i*size : (i+1)*size]
		switch dtype {
		case "F32":
			out[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(chunk)))
		case "I32":
			out[i] = float64(int32(binary.LittleEndian.Uint32(chunk)))
		default:
			return nil, fmt.Errorf("unsupported dtype %s", dtype)
		}
	}
	return out, nil
}

func dtypeSize(dtype string) (int, error) {
	switch dtype {
	case "F32", "I32":
		return 4, nil
	default:
		return 0, fmt.Errorf("unsupported dtype %s", dtype)
	}
}

func numel(shape []int) int {
	if len(shape) == 0 {
		return 1
	}
	count := 1
	for _, dim := range shape {
		count *= dim
	}
	return count
}

// Describe returns one line per tensor, in the order the file stores them, with
// up to 16 values each. The tester logs these lines so a learner can see the
// weights it wrote.
func Describe(file []byte) ([]string, error) {
	header, data, err := parseFile(file)
	if err != nil {
		return nil, err
	}
	type entry struct {
		name  string
		dtype string
		shape []int
		start int
		end   int
	}
	var entries []entry
	for name, raw := range header {
		if name == "__metadata__" {
			continue
		}
		var info struct {
			DType       string `json:"dtype"`
			Shape       []int  `json:"shape"`
			DataOffsets [2]int `json:"data_offsets"`
		}
		if err := json.Unmarshal(raw, &info); err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		if info.DataOffsets[0] < 0 || info.DataOffsets[1] > len(data) || info.DataOffsets[0] > info.DataOffsets[1] {
			return nil, fmt.Errorf("%s: data_offsets %v are outside the data section", name, info.DataOffsets)
		}
		entries = append(entries, entry{name, info.DType, info.Shape, info.DataOffsets[0], info.DataOffsets[1]})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].start < entries[j].start })

	const limit = 16
	lines := make([]string, 0, len(entries))
	for _, e := range entries {
		values, err := firstValues(e.dtype, data[e.start:e.end], limit)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", e.name, err)
		}
		text := fmt.Sprint(values)
		if numel(e.shape) > limit {
			text = text[:len(text)-1] + " ...]"
		}
		lines = append(lines, fmt.Sprintf("%s %s %v: %s", e.name, e.dtype, e.shape, text))
	}
	return lines, nil
}
