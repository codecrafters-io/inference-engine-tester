package internal

import "github.com/codecrafters-io/inference-engine-tester/internal/safetensors"

type namedTensor struct {
	name   string
	shape  []int
	values []float64
}

func tensorsFrom(parts []namedTensor) []safetensors.Tensor {
	out := make([]safetensors.Tensor, len(parts))
	for i, part := range parts {
		out[i] = f32Tensor(part.name, part.shape, part.values...)
	}
	return out
}

func zeros(n int) []float64 {
	return make([]float64, n)
}

func ones(n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = 1
	}
	return out
}

func zeroLayer(layer, nEmbd int) []namedTensor {
	inner := 4 * nEmbd
	width := 3 * nEmbd
	prefix := "h." + itoa(layer)
	return []namedTensor{
		{prefix + ".ln_1.weight", []int{nEmbd}, ones(nEmbd)},
		{prefix + ".ln_1.bias", []int{nEmbd}, zeros(nEmbd)},
		{prefix + ".attn.c_attn.weight", []int{nEmbd, width}, zeros(nEmbd * width)},
		{prefix + ".attn.c_attn.bias", []int{width}, zeros(width)},
		{prefix + ".attn.c_proj.weight", []int{nEmbd, nEmbd}, zeros(nEmbd * nEmbd)},
		{prefix + ".attn.c_proj.bias", []int{nEmbd}, zeros(nEmbd)},
		{prefix + ".ln_2.weight", []int{nEmbd}, ones(nEmbd)},
		{prefix + ".ln_2.bias", []int{nEmbd}, zeros(nEmbd)},
		{prefix + ".mlp.c_fc.weight", []int{nEmbd, inner}, zeros(nEmbd * inner)},
		{prefix + ".mlp.c_fc.bias", []int{inner}, zeros(inner)},
		{prefix + ".mlp.c_proj.weight", []int{inner, nEmbd}, zeros(inner * nEmbd)},
		{prefix + ".mlp.c_proj.bias", []int{nEmbd}, zeros(nEmbd)},
	}
}

// residualLayer is zeroLayer with attention that copies each row into its
// query, key, and value and projects it back unchanged. The block's output
// then depends on the layer norm and on the residual add, so a program that
// echoes its input, or adds onto the normalized copy, gets a different answer.
// zeroLayer alone cannot show any of that: with every attention tensor zero,
// a correct block returns its input.
func residualLayer(layer, nEmbd int) []namedTensor {
	parts := zeroLayer(layer, nEmbd)
	for _, attn := range identityQKV(layer, nEmbd) {
		for i := range parts {
			if parts[i].name == attn.name {
				parts[i] = attn
			}
		}
	}
	return parts
}

// withMLPBump makes the MLP read the row it is given: the wide row keeps the
// first two numbers of the normalized row, GELU applies to both, and the
// projection puts them back. One number is negative and one positive, so a
// wrong GELU shows up either way, and skipping ln_2 changes both.
func withMLPBump(layer, nEmbd int) []namedTensor {
	parts := residualLayer(layer, nEmbd)
	inner := 4 * nEmbd
	fc := zeros(nEmbd * inner)
	proj := zeros(inner * nEmbd)
	for i := 0; i < nEmbd; i++ {
		fc[i*inner+i] = 1
		proj[i*nEmbd+i] = 1
	}
	prefix := "h." + itoa(layer)
	for i := range parts {
		switch parts[i].name {
		case prefix + ".mlp.c_fc.weight":
			parts[i].values = fc
		case prefix + ".mlp.c_proj.weight":
			parts[i].values = proj
		}
	}
	return parts
}

func withGeluBump(layer int, parts []namedTensor) []namedTensor {
	inner := 8
	bias := zeros(inner)
	bias[0] = 1
	proj := zeros(inner * 2)
	proj[0] = 1
	prefix := "h." + itoa(layer)
	for i := range parts {
		if parts[i].name == prefix+".mlp.c_fc.bias" {
			parts[i].values = bias
		}
		if parts[i].name == prefix+".mlp.c_proj.weight" {
			parts[i].values = proj
		}
	}
	return parts
}

func identityQKV(layer, nEmbd int) []namedTensor {
	width := 3 * nEmbd
	weight := make([]float64, nEmbd*width)
	for i := 0; i < nEmbd; i++ {
		weight[i*width+i] = 1
		weight[i*width+nEmbd+i] = 1
		weight[i*width+2*nEmbd+i] = 1
	}
	proj := make([]float64, nEmbd*nEmbd)
	for i := 0; i < nEmbd; i++ {
		proj[i*nEmbd+i] = 1
	}
	prefix := "h." + itoa(layer) + ".attn"
	return []namedTensor{
		{prefix + ".c_attn.weight", []int{nEmbd, width}, weight},
		{prefix + ".c_attn.bias", []int{width}, zeros(width)},
		{prefix + ".c_proj.weight", []int{nEmbd, nEmbd}, proj},
		{prefix + ".c_proj.bias", []int{nEmbd}, zeros(nEmbd)},
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var digits []byte
	for n > 0 {
		digits = append([]byte{byte('0' + n%10)}, digits...)
		n /= 10
	}
	return string(digits)
}
