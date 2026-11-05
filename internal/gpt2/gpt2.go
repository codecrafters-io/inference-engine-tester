// Package gpt2 is the tester's oracle for GPT-2's forward pass.
//
// The formulas match the published GPT-2 config: gelu_new, layer-norm epsilon
// 1e-5, attention scores divided by sqrt(head size), and logits tied to the
// token embedding. Future positions are dropped before the softmax.
package gpt2

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/codecrafters-io/inference-engine-tester/internal/safetensors"
)

const lnEps = 1e-5

// Config is the subset of GPT-2's config.json the forward pass reads.
type Config struct {
	NEmbd      int `json:"n_embd"`
	NHead      int `json:"n_head"`
	NLayer     int `json:"n_layer"`
	NPositions int `json:"n_positions"`
	VocabSize  int `json:"vocab_size"`
}

// Weights holds the tensors from one safetensors file.
type Weights struct {
	tensors map[string]tensor
}

type tensor struct {
	shape []int
	data  []float64
}

// Load reads an F32 safetensors file.
func Load(file []byte) (*Weights, error) {
	raw, err := safetensors.Read(file)
	if err != nil {
		return nil, err
	}
	out := &Weights{tensors: make(map[string]tensor, len(raw))}
	for _, item := range raw {
		values, err := item.Floats()
		if err != nil {
			return nil, err
		}
		out.tensors[item.Name] = tensor{shape: item.Shape, data: values}
	}
	return out, nil
}

// Embed returns one row per token id: the token row plus the position row.
// Position 0 is the first id.
func Embed(w *Weights, ids []int) ([][]float64, error) {
	wte, err := w.matrix("wte.weight")
	if err != nil {
		return nil, err
	}
	wpe, err := w.matrix("wpe.weight")
	if err != nil {
		return nil, err
	}
	out := make([][]float64, len(ids))
	for i, id := range ids {
		if id < 0 || id >= len(wte) {
			return nil, fmt.Errorf("token id %d is outside wte.weight", id)
		}
		if i >= len(wpe) {
			return nil, fmt.Errorf("position %d is outside wpe.weight", i)
		}
		row := make([]float64, len(wte[id]))
		for j := range row {
			row[j] = wte[id][j] + wpe[i][j]
		}
		out[i] = row
	}
	return out, nil
}

// Attention runs causal multi-head self-attention for one layer, including the
// output projection. x is one row per position.
func Attention(w *Weights, cfg Config, layer int, x [][]float64) ([][]float64, error) {
	if cfg.NHead <= 0 || cfg.NEmbd%cfg.NHead != 0 {
		return nil, fmt.Errorf("n_embd %d is not divisible by n_head %d", cfg.NEmbd, cfg.NHead)
	}
	weight, err := w.matrix(fmt.Sprintf("h.%d.attn.c_attn.weight", layer))
	if err != nil {
		return nil, err
	}
	bias, err := w.vector(fmt.Sprintf("h.%d.attn.c_attn.bias", layer))
	if err != nil {
		return nil, err
	}
	projW, err := w.matrix(fmt.Sprintf("h.%d.attn.c_proj.weight", layer))
	if err != nil {
		return nil, err
	}
	projB, err := w.vector(fmt.Sprintf("h.%d.attn.c_proj.bias", layer))
	if err != nil {
		return nil, err
	}
	qkv := addBias(matmul(x, weight), bias)
	headDim := cfg.NEmbd / cfg.NHead
	scale := math.Sqrt(float64(headDim))
	seq := len(x)
	concat := make([][]float64, seq)
	for i := range concat {
		concat[i] = make([]float64, cfg.NEmbd)
	}
	for head := 0; head < cfg.NHead; head++ {
		qBase := head * headDim
		kBase := cfg.NEmbd + qBase
		vBase := 2*cfg.NEmbd + qBase
		for i := 0; i < seq; i++ {
			scores := make([]float64, i+1)
			for j := 0; j <= i; j++ {
				var dot float64
				for d := 0; d < headDim; d++ {
					dot += qkv[i][qBase+d] * qkv[j][kBase+d]
				}
				scores[j] = dot / scale
			}
			probs := softmax(scores)
			for d := 0; d < headDim; d++ {
				var sum float64
				for j := 0; j <= i; j++ {
					sum += probs[j] * qkv[j][vBase+d]
				}
				concat[i][qBase+d] = sum
			}
		}
	}
	return addBias(matmul(concat, projW), projB), nil
}

// Block runs one transformer block: layer norm, attention, residual, layer
// norm, MLP, residual.
func Block(w *Weights, cfg Config, layer int, x [][]float64) ([][]float64, error) {
	normed, err := layernormRows(w, x, fmt.Sprintf("h.%d.ln_1", layer))
	if err != nil {
		return nil, err
	}
	attn, err := Attention(w, cfg, layer, normed)
	if err != nil {
		return nil, err
	}
	y := addRows(x, attn)
	normed, err = layernormRows(w, y, fmt.Sprintf("h.%d.ln_2", layer))
	if err != nil {
		return nil, err
	}
	mlp, err := mlp(w, normed, layer)
	if err != nil {
		return nil, err
	}
	return addRows(y, mlp), nil
}

// Logits runs every layer and returns the scores for the last position.
// Score i is the dot product of that position with row i of wte.weight.
func Logits(w *Weights, cfg Config, ids []int) ([]float64, error) {
	hidden, err := forward(w, cfg, ids)
	if err != nil {
		return nil, err
	}
	last, err := layernorm(w, hidden[len(hidden)-1], "ln_f")
	if err != nil {
		return nil, err
	}
	wte, err := w.matrix("wte.weight")
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(wte))
	for i, row := range wte {
		var dot float64
		for j := range last {
			dot += last[j] * row[j]
		}
		out[i] = dot
	}
	return out, nil
}

// Generate appends count tokens, each the highest-scoring id. Ties keep the
// lower id. The returned ids are only the new tokens.
func Generate(w *Weights, cfg Config, ids []int, count int) ([]int, error) {
	return generate(w, cfg, ids, count, 0, 0, nil)
}

// Sample appends count tokens. Each step divides the logits by temperature,
// applies softmax, keeps the smallest highest-probability set whose mass
// reaches topP, renormalizes, and walks that cumulative mass until it exceeds
// the matching random number. Ties in the sort keep the lower id.
func Sample(w *Weights, cfg Config, ids []int, count int, temperature, topP float64, random []float64) ([]int, error) {
	if len(random) != count {
		return nil, fmt.Errorf("expected %d random numbers, got %d", count, len(random))
	}
	return generate(w, cfg, ids, count, temperature, topP, random)
}

func generate(w *Weights, cfg Config, prompt []int, count int, temperature, topP float64, random []float64) ([]int, error) {
	cache := newCache(cfg)
	ids := append([]int(nil), prompt...)
	hidden, err := Embed(w, ids)
	if err != nil {
		return nil, err
	}
	for layer := 0; layer < cfg.NLayer; layer++ {
		hidden, err = blockCached(w, cfg, layer, hidden, &cache)
		if err != nil {
			return nil, err
		}
	}
	out := make([]int, 0, count)
	for step := 0; step < count; step++ {
		next, err := pick(w, hidden[len(hidden)-1], temperature, topP, randomValue(random, step))
		if err != nil {
			return nil, err
		}
		out = append(out, next)
		ids = append(ids, next)
		row, err := Embed(w, ids[len(ids)-1:])
		if err != nil {
			return nil, err
		}
		// Embed uses position 0 for a one-id slice. The new token's position is
		// its index in the full sequence.
		wpe, err := w.matrix("wpe.weight")
		if err != nil {
			return nil, err
		}
		wte, err := w.matrix("wte.weight")
		if err != nil {
			return nil, err
		}
		pos := len(ids) - 1
		row[0] = make([]float64, len(wte[next]))
		for j := range row[0] {
			row[0][j] = wte[next][j] + wpe[pos][j]
		}
		for layer := 0; layer < cfg.NLayer; layer++ {
			row, err = blockCached(w, cfg, layer, row, &cache)
			if err != nil {
				return nil, err
			}
		}
		hidden = row
	}
	return out, nil
}

func randomValue(random []float64, step int) float64 {
	if random == nil {
		return 0
	}
	return random[step]
}

func pick(w *Weights, last []float64, temperature, topP float64, random float64) (int, error) {
	scored, err := layernorm(w, last, "ln_f")
	if err != nil {
		return 0, err
	}
	wte, err := w.matrix("wte.weight")
	if err != nil {
		return 0, err
	}
	if temperature == 0 {
		best := 0
		bestScore := -math.MaxFloat64
		for i, row := range wte {
			var dot float64
			for j := range scored {
				dot += scored[j] * row[j]
			}
			if dot > bestScore {
				best = i
				bestScore = dot
			}
		}
		return best, nil
	}
	logits := make([]float64, len(wte))
	for i, row := range wte {
		var dot float64
		for j := range scored {
			dot += scored[j] * row[j]
		}
		logits[i] = dot / temperature
	}
	return sampleToken(logits, topP, random), nil
}

func sampleToken(logits []float64, topP, random float64) int {
	probs := softmax(logits)
	order := make([]int, len(probs))
	for i := range order {
		order[i] = i
	}
	// Highest probability first. Equal probabilities keep the lower id, and
	// order starts sorted by id, so a strict greater-probability swap does that.
	for i := 1; i < len(order); i++ {
		for j := i; j > 0 && probs[order[j]] > probs[order[j-1]]; j-- {
			order[j], order[j-1] = order[j-1], order[j]
		}
	}
	kept := make([]int, 0, len(order))
	var mass float64
	for _, id := range order {
		kept = append(kept, id)
		mass += probs[id]
		if mass >= topP {
			break
		}
	}
	var keptMass float64
	for _, id := range kept {
		keptMass += probs[id]
	}
	var cum float64
	for _, id := range kept {
		cum += probs[id] / keptMass
		if cum > random {
			return id
		}
	}
	return kept[len(kept)-1]
}

type cache struct {
	keys   [][][][]float64
	values [][][][]float64
}

func newCache(cfg Config) cache {
	keys := make([][][][]float64, cfg.NLayer)
	values := make([][][][]float64, cfg.NLayer)
	for layer := 0; layer < cfg.NLayer; layer++ {
		keys[layer] = make([][][]float64, cfg.NHead)
		values[layer] = make([][][]float64, cfg.NHead)
	}
	return cache{keys: keys, values: values}
}

func blockCached(w *Weights, cfg Config, layer int, x [][]float64, c *cache) ([][]float64, error) {
	normed, err := layernormRows(w, x, fmt.Sprintf("h.%d.ln_1", layer))
	if err != nil {
		return nil, err
	}
	attn, err := attentionCached(w, cfg, layer, normed, c)
	if err != nil {
		return nil, err
	}
	y := addRows(x, attn)
	normed, err = layernormRows(w, y, fmt.Sprintf("h.%d.ln_2", layer))
	if err != nil {
		return nil, err
	}
	projected, err := mlp(w, normed, layer)
	if err != nil {
		return nil, err
	}
	return addRows(y, projected), nil
}

func attentionCached(w *Weights, cfg Config, layer int, x [][]float64, c *cache) ([][]float64, error) {
	weight, err := w.matrix(fmt.Sprintf("h.%d.attn.c_attn.weight", layer))
	if err != nil {
		return nil, err
	}
	bias, err := w.vector(fmt.Sprintf("h.%d.attn.c_attn.bias", layer))
	if err != nil {
		return nil, err
	}
	projW, err := w.matrix(fmt.Sprintf("h.%d.attn.c_proj.weight", layer))
	if err != nil {
		return nil, err
	}
	projB, err := w.vector(fmt.Sprintf("h.%d.attn.c_proj.bias", layer))
	if err != nil {
		return nil, err
	}
	qkv := addBias(matmul(x, weight), bias)
	headDim := cfg.NEmbd / cfg.NHead
	scale := math.Sqrt(float64(headDim))
	concat := make([][]float64, len(x))
	for i := range concat {
		concat[i] = make([]float64, cfg.NEmbd)
	}
	for head := 0; head < cfg.NHead; head++ {
		qBase := head * headDim
		kBase := cfg.NEmbd + qBase
		vBase := 2*cfg.NEmbd + qBase
		for i := range x {
			key := make([]float64, headDim)
			value := make([]float64, headDim)
			for d := 0; d < headDim; d++ {
				key[d] = qkv[i][kBase+d]
				value[d] = qkv[i][vBase+d]
			}
			c.keys[layer][head] = append(c.keys[layer][head], key)
			c.values[layer][head] = append(c.values[layer][head], value)
			scores := make([]float64, len(c.keys[layer][head]))
			for j, past := range c.keys[layer][head] {
				var dot float64
				for d := 0; d < headDim; d++ {
					dot += qkv[i][qBase+d] * past[d]
				}
				scores[j] = dot / scale
			}
			probs := softmax(scores)
			for d := 0; d < headDim; d++ {
				var sum float64
				for j, prob := range probs {
					sum += prob * c.values[layer][head][j][d]
				}
				concat[i][qBase+d] = sum
			}
		}
	}
	return addBias(matmul(concat, projW), projB), nil
}

func forward(w *Weights, cfg Config, ids []int) ([][]float64, error) {
	hidden, err := Embed(w, ids)
	if err != nil {
		return nil, err
	}
	for layer := 0; layer < cfg.NLayer; layer++ {
		hidden, err = Block(w, cfg, layer, hidden)
		if err != nil {
			return nil, err
		}
	}
	return hidden, nil
}

func mlp(w *Weights, x [][]float64, layer int) ([][]float64, error) {
	fcW, err := w.matrix(fmt.Sprintf("h.%d.mlp.c_fc.weight", layer))
	if err != nil {
		return nil, err
	}
	fcB, err := w.vector(fmt.Sprintf("h.%d.mlp.c_fc.bias", layer))
	if err != nil {
		return nil, err
	}
	projW, err := w.matrix(fmt.Sprintf("h.%d.mlp.c_proj.weight", layer))
	if err != nil {
		return nil, err
	}
	projB, err := w.vector(fmt.Sprintf("h.%d.mlp.c_proj.bias", layer))
	if err != nil {
		return nil, err
	}
	hidden := addBias(matmul(x, fcW), fcB)
	for i := range hidden {
		for j := range hidden[i] {
			hidden[i][j] = gelu(hidden[i][j])
		}
	}
	return addBias(matmul(hidden, projW), projB), nil
}

func gelu(x float64) float64 {
	return 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x)))
}

func layernormRows(w *Weights, rows [][]float64, prefix string) ([][]float64, error) {
	out := make([][]float64, len(rows))
	for i, row := range rows {
		normed, err := layernorm(w, row, prefix)
		if err != nil {
			return nil, err
		}
		out[i] = normed
	}
	return out, nil
}

func layernorm(w *Weights, row []float64, prefix string) ([]float64, error) {
	weight, err := w.vector(prefix + ".weight")
	if err != nil {
		return nil, err
	}
	bias, err := w.vector(prefix + ".bias")
	if err != nil {
		return nil, err
	}
	n := float64(len(row))
	var mean float64
	for _, value := range row {
		mean += value
	}
	mean /= n
	var variance float64
	for _, value := range row {
		d := value - mean
		variance += d * d
	}
	variance /= n
	inv := 1 / math.Sqrt(variance+lnEps)
	out := make([]float64, len(row))
	for i, value := range row {
		out[i] = (value-mean)*inv*weight[i] + bias[i]
	}
	return out, nil
}

func (w *Weights) matrix(name string) ([][]float64, error) {
	item, ok := w.tensors[name]
	if !ok {
		return nil, fmt.Errorf("missing tensor %s", name)
	}
	if len(item.shape) != 2 {
		return nil, fmt.Errorf("%s: expected a matrix, got shape %v", name, item.shape)
	}
	rows, cols := item.shape[0], item.shape[1]
	out := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		out[i] = item.data[i*cols : (i+1)*cols]
	}
	return out, nil
}

func (w *Weights) vector(name string) ([]float64, error) {
	item, ok := w.tensors[name]
	if !ok {
		return nil, fmt.Errorf("missing tensor %s", name)
	}
	if len(item.shape) != 1 {
		return nil, fmt.Errorf("%s: expected a vector, got shape %v", name, item.shape)
	}
	return item.data, nil
}

func matmul(a, b [][]float64) [][]float64 {
	out := make([][]float64, len(a))
	for i, row := range a {
		out[i] = make([]float64, len(b[0]))
		for k, value := range row {
			bk := b[k]
			for j := range out[i] {
				out[i][j] += value * bk[j]
			}
		}
	}
	return out
}

func addBias(rows [][]float64, bias []float64) [][]float64 {
	for i := range rows {
		for j := range rows[i] {
			rows[i][j] += bias[j]
		}
	}
	return rows
}

func addRows(a, b [][]float64) [][]float64 {
	out := make([][]float64, len(a))
	for i := range a {
		out[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			out[i][j] = a[i][j] + b[i][j]
		}
	}
	return out
}

func softmax(values []float64) []float64 {
	max := values[0]
	for _, value := range values[1:] {
		if value > max {
			max = value
		}
	}
	out := make([]float64, len(values))
	var sum float64
	for i, value := range values {
		out[i] = math.Exp(value - max)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// ParseConfig reads the config.json the tester writes next to the weights.
func ParseConfig(raw []byte) (Config, error) {
	var cfg Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return Config{}, err
	}
	return cfg, nil
}
