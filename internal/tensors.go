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
