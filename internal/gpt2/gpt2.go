// Package gpt2 is the tester's oracle for GPT-2's forward pass.
//
// The formulas match the published GPT-2 config: gelu_new, layer-norm epsilon
// 1e-5, attention scores divided by sqrt(head size), and logits tied to the
// token embedding. Future positions are dropped before the softmax.
package gpt2

import (
	"cmp"
	"encoding/json"
	"fmt"
	"math"
	"slices"

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
	bug     Bug
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
	out := make([][]float64, len(ids))
	for i, id := range ids {
		tokenRow, positionRow := id, i
		switch w.bug {
		case EmbedPositionByID:
			positionRow = id
		case EmbedTokenByIndex:
			tokenRow = i
		}
		row, err := embedRow(w, tokenRow, positionRow)
		if err != nil {
			return nil, err
		}
		out[i] = row
	}
	return out, nil
}

func embedRow(w *Weights, id, position int) ([]float64, error) {
	wte, err := w.matrix("wte.weight")
	if err != nil {
		return nil, err
	}
	wpe, err := w.matrix("wpe.weight")
	if err != nil {
		return nil, err
	}
	if id < 0 || id >= len(wte) {
		return nil, fmt.Errorf("token id %d is outside wte.weight", id)
	}
	if position < 0 || position >= len(wpe) {
		return nil, fmt.Errorf("position %d is outside wpe.weight", position)
	}
	row := make([]float64, len(wte[id]))
	for j := range row {
		row[j] = wte[id][j]
		if w.bug != EmbedNoPosition {
			row[j] += wpe[position][j]
		}
	}
	return row, nil
}

// heads is one layer's queries, keys, and values for some rows, and how they
// split into heads.
type heads struct {
	qkv     [][]float64
	nEmbd   int
	nHead   int
	headDim int
	scale   float64
}

func (h heads) bases(head int) (q, k, v int) {
	q = head * h.headDim
	return q, h.nEmbd + q, 2*h.nEmbd + q
}

func splitHeads(w *Weights, cfg Config, layer int, x [][]float64) (heads, error) {
	if cfg.NHead <= 0 || cfg.NEmbd%cfg.NHead != 0 {
		return heads{}, fmt.Errorf("n_embd %d is not divisible by n_head %d", cfg.NEmbd, cfg.NHead)
	}
	weight, err := w.matrix(fmt.Sprintf("h.%d.attn.c_attn.weight", layer))
	if err != nil {
		return heads{}, err
	}
	bias, err := w.vector(fmt.Sprintf("h.%d.attn.c_attn.bias", layer))
	if err != nil {
		return heads{}, err
	}
	n := cfg.NEmbd
	var qkv [][]float64
	switch w.bug {
	case AttnNoWeights:
		qkv = make([][]float64, len(x))
		for i, row := range x {
			qkv[i] = slices.Concat(row, row, row)
		}
	case AttnNoBias:
		qkv = matmul(x, weight)
	default:
		qkv = addBias(matmul(x, weight), bias)
	}
	if w.bug == AttnSwapQK {
		for _, row := range qkv {
			for d := 0; d < n; d++ {
				row[d], row[n+d] = row[n+d], row[d]
			}
		}
	}
	out := heads{qkv: qkv, nEmbd: n, nHead: cfg.NHead}
	if w.bug == AttnOneHead {
		out.nHead = 1
	}
	out.headDim = n / out.nHead
	switch w.bug {
	case AttnNoScale:
		out.scale = 1
	case AttnScaleByEmbd:
		out.scale = math.Sqrt(float64(n))
	default:
		out.scale = math.Sqrt(float64(out.headDim))
	}
	return out, nil
}

func projectHeads(w *Weights, layer int, concat [][]float64) ([][]float64, error) {
	if w.bug == AttnNoProj {
		return concat, nil
	}
	projW, err := w.matrix(fmt.Sprintf("h.%d.attn.c_proj.weight", layer))
	if err != nil {
		return nil, err
	}
	projB, err := w.vector(fmt.Sprintf("h.%d.attn.c_proj.bias", layer))
	if err != nil {
		return nil, err
	}
	out := matmul(concat, projW)
	if w.bug == AttnNoProjBias {
		return out, nil
	}
	return addBias(out, projB), nil
}

// Attention runs causal multi-head self-attention for one layer, including the
// output projection. x is one row per position.
func Attention(w *Weights, cfg Config, layer int, x [][]float64) ([][]float64, error) {
	if w.bug == LayerZero {
		layer = 0
	}
	h, err := splitHeads(w, cfg, layer, x)
	if err != nil {
		return nil, err
	}
	seq := len(x)
	concat := make([][]float64, seq)
	for i := range concat {
		concat[i] = make([]float64, cfg.NEmbd)
	}
	for head := 0; head < h.nHead; head++ {
		qBase, kBase, vBase := h.bases(head)
		for i := 0; i < seq; i++ {
			visible := i + 1
			if w.bug == AttnNoMask {
				visible = seq
			}
			scores := make([]float64, visible)
			for j := range scores {
				var dot float64
				for d := 0; d < h.headDim; d++ {
					dot += h.qkv[i][qBase+d] * h.qkv[j][kBase+d]
				}
				scores[j] = dot / h.scale
			}
			probs := softmax(scores)
			for d := 0; d < h.headDim; d++ {
				var sum float64
				for j, prob := range probs {
					sum += prob * h.qkv[j][vBase+d]
				}
				concat[i][qBase+d] = sum
			}
		}
	}
	return projectHeads(w, layer, concat)
}

// Block runs one transformer block: layer norm, attention, residual, layer
// norm, MLP, residual.
func Block(w *Weights, cfg Config, layer int, x [][]float64) ([][]float64, error) {
	return block(w, layer, x, func(layer int, normed [][]float64) ([][]float64, error) {
		return Attention(w, cfg, layer, normed)
	})
}

type attendFunc func(layer int, normed [][]float64) ([][]float64, error)

func block(w *Weights, layer int, x [][]float64, attend attendFunc) ([][]float64, error) {
	if w.bug == LayerZero {
		layer = 0
	}
	ln1 := fmt.Sprintf("h.%d.ln_1", layer)
	ln2 := fmt.Sprintf("h.%d.ln_2", layer)

	normed := x
	var err error
	if w.bug != BlockNoLn1 {
		if normed, err = layernormRows(w, x, ln1); err != nil {
			return nil, err
		}
	}
	attn, err := attend(layer, normed)
	if err != nil {
		return nil, err
	}
	var y [][]float64
	switch w.bug {
	case BlockAddToNormed:
		y = addRows(normed, attn)
	case BlockNoResidual:
		y = attn
	default:
		y = addRows(x, attn)
	}

	var copies [][]float64
	switch w.bug {
	case BlockNoLn2:
		copies = y
	case BlockLn1ForMLP:
		copies, err = layernormRows(w, y, ln1)
	case MlpFromInput:
		copies, err = layernormRows(w, x, ln2)
	default:
		copies, err = layernormRows(w, y, ln2)
	}
	if err != nil {
		return nil, err
	}
	projected, err := mlp(w, copies, layer)
	if err != nil {
		return nil, err
	}
	switch w.bug {
	case MlpNoResidual:
		return projected, nil
	case MlpOntoNormed:
		return addRows(copies, projected), nil
	}
	return addRows(y, projected), nil
}

// Logits runs every layer and returns the scores for the last position.
// Score i is the dot product of that position with row i of wte.weight.
func Logits(w *Weights, cfg Config, ids []int) ([]float64, error) {
	hidden, err := forward(w, cfg, ids)
	if err != nil {
		return nil, err
	}
	row := hidden[len(hidden)-1]
	if w.bug == LogitsFirstRow {
		row = hidden[0]
	}
	return score(w, row)
}

// score layer-normalizes one row with ln_f and scores every token against it.
func score(w *Weights, row []float64) ([]float64, error) {
	last := row
	if w.bug != LogitsNoLnF {
		var err error
		if last, err = layernorm(w, row, "ln_f"); err != nil {
			return nil, err
		}
	}
	wte, err := w.matrix("wte.weight")
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(wte))
	for i, tokenRow := range wte {
		var dot float64
		for j := range last {
			dot += last[j] * tokenRow[j]
		}
		out[i] = dot
	}
	return out, nil
}

// Step is one new token and the logits it was picked from.
type Step struct {
	Logits []float64
	Token  int
}

// Generate appends count tokens, each the highest-scoring id. Ties keep the
// lower id. The returned ids are only the new tokens.
func Generate(w *Weights, cfg Config, ids []int, count int) ([]int, error) {
	return tokens(Steps(w, cfg, ids, count, 0, 0, nil))
}

// Sample appends count tokens. Each step divides the logits by temperature,
// applies softmax, keeps the smallest highest-probability set whose mass
// reaches topP, renormalizes, and walks that cumulative mass until it exceeds
// the matching random number. Ties in the sort keep the lower id.
func Sample(w *Weights, cfg Config, ids []int, count int, temperature, topP float64, random []float64) ([]int, error) {
	if len(random) != count {
		return nil, fmt.Errorf("expected %d random numbers, got %d", count, len(random))
	}
	return tokens(Steps(w, cfg, ids, count, temperature, topP, random))
}

func tokens(steps []Step, err error) ([]int, error) {
	if err != nil {
		return nil, err
	}
	out := make([]int, len(steps))
	for i, step := range steps {
		out[i] = step.Token
	}
	return out, nil
}

// Steps generates count tokens and keeps the logits behind each one. A nil
// random means greedy decoding.
func Steps(w *Weights, cfg Config, prompt []int, count int, temperature, topP float64, random []float64) ([]Step, error) {
	c := newCache(cfg)
	attend := func(layer int, normed [][]float64) ([][]float64, error) {
		return attentionCached(w, cfg, layer, normed, &c)
	}
	hidden, err := Embed(w, prompt)
	if err != nil {
		return nil, err
	}
	for layer := 0; layer < cfg.NLayer; layer++ {
		if hidden, err = block(w, layer, hidden, attend); err != nil {
			return nil, err
		}
	}
	if w.bug == CacheDropsPrompt {
		c = newCache(cfg)
	}
	out := make([]Step, 0, count)
	for step := 0; step < count; step++ {
		if w.bug == GenNoFeedback && step > 0 {
			out = append(out, out[0])
			continue
		}
		logits, err := score(w, hidden[len(hidden)-1])
		if err != nil {
			return nil, err
		}
		var next int
		if random == nil || w.bug == SampleGreedy {
			next = argmax(logits, w.bug)
		} else {
			r := random[step]
			if w.bug == SampleReuseRandom {
				r = random[0]
			}
			next = sampleToken(logits, temperature, topP, r, w.bug)
		}
		out = append(out, Step{Logits: logits, Token: next})
		// The new token's position is its index in the whole sequence.
		position := len(prompt) + step
		switch w.bug {
		case CachePositionZero:
			position = 0
		case CachePositionAmongNew:
			position = step
		}
		row, err := embedRow(w, next, position)
		if err != nil {
			return nil, err
		}
		hidden = [][]float64{row}
		for layer := 0; layer < cfg.NLayer; layer++ {
			if hidden, err = block(w, layer, hidden, attend); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

func argmax(logits []float64, bug Bug) int {
	best := 0
	for i, value := range logits {
		if value > logits[best] || (bug == PickTieHigher && value == logits[best]) {
			best = i
		}
	}
	return best
}

// sampleOrder returns the probabilities after temperature and the token ids
// sorted by probability, highest first. Equal probabilities keep the lower id.
func sampleOrder(logits []float64, temperature float64, bug Bug) ([]float64, []int) {
	scaled := logits
	if bug != SampleNoTemperature {
		scaled = make([]float64, len(logits))
		for i, value := range logits {
			scaled[i] = value / temperature
		}
	}
	probs := softmax(scaled)
	order := make([]int, len(probs))
	for i := range order {
		order[i] = i
	}
	if bug != SampleIDOrder {
		slices.SortStableFunc(order, func(a, b int) int { return cmp.Compare(probs[b], probs[a]) })
	}
	return probs, order
}

func sampleToken(logits []float64, temperature, topP, random float64, bug Bug) int {
	probs, order := sampleOrder(logits, temperature, bug)
	if bug == SampleNoTopP {
		topP = 1
	}
	kept := make([]int, 0, len(order))
	var mass float64
	for _, id := range order {
		if bug == SampleTopPBefore && len(kept) > 0 && mass+probs[id] >= topP {
			break
		}
		kept = append(kept, id)
		mass += probs[id]
		if mass >= topP {
			break
		}
	}
	keptMass := 1.0
	if bug != SampleNoRenorm {
		keptMass = 0
		for _, id := range kept {
			keptMass += probs[id]
		}
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

// GreedyMargin is the gap between the two highest logits. A wide gap means
// the same token wins whatever float type the learner's engine computes in.
func GreedyMargin(logits []float64) float64 {
	sorted := slices.Clone(logits)
	slices.Sort(sorted)
	if len(sorted) < 2 {
		return math.Inf(1)
	}
	return sorted[len(sorted)-1] - sorted[len(sorted)-2]
}

// SampleMargins is how far one sampling step is from going another way.
// order is the smallest gap between two probabilities next to each other in
// the sorted order. boundary is the smallest gap between a running total and
// top_p, or between a running total of the kept tokens and the random number.
func SampleMargins(logits []float64, temperature, topP, random float64) (order, boundary float64) {
	probs, sorted := sampleOrder(logits, temperature, NoBug)
	order, boundary = math.Inf(1), math.Inf(1)
	kept := len(sorted)
	var mass float64
	for i, id := range sorted {
		if i > 0 {
			order = math.Min(order, probs[sorted[i-1]]-probs[id])
		}
		// A top_p of 1 keeps every token, so no running total is a boundary.
		if topP < 1 && kept == len(sorted) {
			mass += probs[id]
			boundary = math.Min(boundary, math.Abs(mass-topP))
			if mass >= topP {
				kept = i + 1
			}
		}
	}
	var keptMass, cum float64
	for _, id := range sorted[:kept] {
		keptMass += probs[id]
	}
	for _, id := range sorted[:kept-1] {
		cum += probs[id] / keptMass
		boundary = math.Min(boundary, math.Abs(cum-random))
	}
	return order, boundary
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

func attentionCached(w *Weights, cfg Config, layer int, x [][]float64, c *cache) ([][]float64, error) {
	h, err := splitHeads(w, cfg, layer, x)
	if err != nil {
		return nil, err
	}
	slot := layer
	if w.bug == CacheSharedAcrossLayers {
		slot = 0
	}
	concat := make([][]float64, len(x))
	for i := range concat {
		concat[i] = make([]float64, cfg.NEmbd)
	}
	for head := 0; head < h.nHead; head++ {
		qBase, kBase, vBase := h.bases(head)
		for i := range x {
			c.keys[slot][head] = append(c.keys[slot][head], slices.Clone(h.qkv[i][kBase:kBase+h.headDim]))
			c.values[slot][head] = append(c.values[slot][head], slices.Clone(h.qkv[i][vBase:vBase+h.headDim]))
			scores := make([]float64, len(c.keys[slot][head]))
			for j, past := range c.keys[slot][head] {
				var dot float64
				for d := 0; d < h.headDim; d++ {
					dot += h.qkv[i][qBase+d] * past[d]
				}
				scores[j] = dot / h.scale
			}
			probs := softmax(scores)
			for d := 0; d < h.headDim; d++ {
				var sum float64
				for j, prob := range probs {
					sum += prob * c.values[slot][head][j][d]
				}
				concat[i][qBase+d] = sum
			}
		}
	}
	return projectHeads(w, layer, concat)
}

func forward(w *Weights, cfg Config, ids []int) ([][]float64, error) {
	hidden, err := Embed(w, ids)
	if err != nil {
		return nil, err
	}
	layers := cfg.NLayer
	if w.bug == LogitsOneLayer {
		layers = 1
	}
	for layer := 0; layer < layers; layer++ {
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
	hidden := matmul(x, fcW)
	if w.bug != MlpNoFcBias {
		hidden = addBias(hidden, fcB)
	}
	for i := range hidden {
		for j := range hidden[i] {
			switch w.bug {
			case MlpNoGelu:
			case MlpRelu:
				hidden[i][j] = math.Max(0, hidden[i][j])
			default:
				hidden[i][j] = gelu(hidden[i][j])
			}
		}
	}
	out := matmul(hidden, projW)
	if w.bug == MlpNoProjBias {
		return out, nil
	}
	return addBias(out, projB), nil
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
	if w.bug == LnSampleVariance && n > 1 {
		variance /= n - 1
	} else {
		variance /= n
	}
	inv := 1 / math.Sqrt(variance+lnEps)
	out := make([]float64, len(row))
	for i, value := range row {
		out[i] = (value - mean) * inv
		if w.bug != LnNoAffine {
			out[i] = out[i]*weight[i] + bias[i]
		}
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
