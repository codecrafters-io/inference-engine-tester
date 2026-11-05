#!/usr/bin/env python3
"""Reference inference engine.

The formulas match the tester's Go oracle: GPT-2 gelu_new, layer-norm epsilon
1e-5, attention scores divided by sqrt(head size), and logits tied to wte.
"""

import json
import math
import struct
import sys

LN_EPS = 1e-5


def main():
    command = sys.argv[1]
    if command == "inspect":
        inspect(sys.argv[2])
        return
    if command == "embed":
        weights = load(sys.argv[2])
        ids = json.load(open(sys.argv[3]))
        print(json.dumps(embed(weights, ids)))
        return
    if command == "attn":
        weights = load(sys.argv[2])
        config = json.load(open(sys.argv[3]))
        layer = json.load(open(sys.argv[4]))
        rows = json.load(open(sys.argv[5]))
        print(json.dumps(attention(weights, config, layer, rows)))
        return
    if command == "block":
        weights = load(sys.argv[2])
        config = json.load(open(sys.argv[3]))
        layer = json.load(open(sys.argv[4]))
        rows = json.load(open(sys.argv[5]))
        print(json.dumps(block(weights, config, layer, rows)))
        return
    if command == "logits":
        weights = load(sys.argv[2])
        config = json.load(open(sys.argv[3]))
        ids = json.load(open(sys.argv[4]))
        print(json.dumps(logits(weights, config, ids)))
        return
    if command == "generate":
        weights = load(sys.argv[2])
        config = json.load(open(sys.argv[3]))
        ids = json.load(open(sys.argv[4]))
        spec = json.load(open(sys.argv[5]))
        count = spec["count"]
        if "temperature" in spec:
            random = read_random(count)
            print(json.dumps(generate(weights, config, ids, count, spec["temperature"], spec["top_p"], random)))
        else:
            print(json.dumps(generate(weights, config, ids, count, 0, 0, None)))
        return
    print(f"Unknown command {command}", file=sys.stderr)
    raise SystemExit(1)


def inspect(path):
    header, data = read_file(path)
    tensors = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        raw = data[start:end]
        count = 1
        for dim in info["shape"]:
            count *= dim
        take = min(4, count)
        dtype = info["dtype"]
        if dtype == "F32":
            values = list(struct.unpack("<" + "f" * take, raw[: take * 4]))
        elif dtype == "I32":
            values = list(struct.unpack("<" + "i" * take, raw[: take * 4]))
        else:
            print(f"unsupported dtype {dtype}", file=sys.stderr)
            raise SystemExit(1)
        tensors.append({"name": name, "dtype": dtype, "shape": info["shape"], "first": values})
    tensors.sort(key=lambda item: item["name"])
    print(json.dumps(tensors))


def read_random(count):
    text = sys.stdin.read().split()
    if len(text) != count:
        print(f"expected {count} random numbers on stdin, got {len(text)}", file=sys.stderr)
        raise SystemExit(1)
    return [float(item) for item in text]


def load(path):
    header, data = read_file(path)
    weights = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        # A zero-copy view of the file's bytes as 32-bit floats. Unpacking GPT-2's
        # 124 million numbers into Python floats would need about 6 GB.
        weights[name] = (info["shape"], memoryview(data)[start:end].cast("f"))
    return weights


def read_file(path):
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        data = handle.read()
    return header, data


def matrix(weights, name):
    shape, values = weights[name]
    rows, cols = shape
    return [values[i * cols : (i + 1) * cols] for i in range(rows)]


def vector(weights, name):
    return weights[name][1]


def embed(weights, ids):
    wte = matrix(weights, "wte.weight")
    wpe = matrix(weights, "wpe.weight")
    out = []
    for pos, token in enumerate(ids):
        row = wte[token]
        pos_row = wpe[pos]
        out.append([row[i] + pos_row[i] for i in range(len(row))])
    return out


def attention(weights, config, layer, rows):
    return attention_from(weights, config, layer, rows, None)


def attention_from(weights, config, layer, rows, cache):
    n_embd = config["n_embd"]
    n_head = config["n_head"]
    head_dim = n_embd // n_head
    scale = math.sqrt(head_dim)
    weight = matrix(weights, f"h.{layer}.attn.c_attn.weight")
    bias = vector(weights, f"h.{layer}.attn.c_attn.bias")
    proj_w = matrix(weights, f"h.{layer}.attn.c_proj.weight")
    proj_b = vector(weights, f"h.{layer}.attn.c_proj.bias")
    qkv = add_bias(matmul(rows, weight), bias)
    concat = [[0.0] * n_embd for _ in rows]
    for head in range(n_head):
        q_base = head * head_dim
        k_base = n_embd + q_base
        v_base = 2 * n_embd + q_base
        if cache is None:
            for i in range(len(rows)):
                scores = []
                for j in range(i + 1):
                    dot = 0.0
                    for d in range(head_dim):
                        dot += qkv[i][q_base + d] * qkv[j][k_base + d]
                    scores.append(dot / scale)
                probs = softmax(scores)
                for d in range(head_dim):
                    total = 0.0
                    for j, prob in enumerate(probs):
                        total += prob * qkv[j][v_base + d]
                    concat[i][q_base + d] = total
        else:
            keys = cache[layer][head]
            for i in range(len(rows)):
                key = [qkv[i][k_base + d] for d in range(head_dim)]
                value = [qkv[i][v_base + d] for d in range(head_dim)]
                keys.append((key, value))
                scores = []
                for past, _value in keys:
                    dot = 0.0
                    for d in range(head_dim):
                        dot += qkv[i][q_base + d] * past[d]
                    scores.append(dot / scale)
                probs = softmax(scores)
                for d in range(head_dim):
                    total = 0.0
                    for j, prob in enumerate(probs):
                        total += prob * keys[j][1][d]
                    concat[i][q_base + d] = total
    return add_bias(matmul(concat, proj_w), proj_b)


def block(weights, config, layer, rows):
    return block_from(weights, config, layer, rows, None)


def block_from(weights, config, layer, rows, cache):
    normed = layer_norm_rows(weights, rows, f"h.{layer}.ln_1")
    attended = attention_from(weights, config, layer, normed, cache)
    hidden = add_rows(rows, attended)
    normed = layer_norm_rows(weights, hidden, f"h.{layer}.ln_2")
    return add_rows(hidden, mlp(weights, normed, layer))


def mlp(weights, rows, layer):
    hidden = add_bias(matmul(rows, matrix(weights, f"h.{layer}.mlp.c_fc.weight")), vector(weights, f"h.{layer}.mlp.c_fc.bias"))
    for row in hidden:
        for i, value in enumerate(row):
            row[i] = gelu(value)
    return add_bias(matmul(hidden, matrix(weights, f"h.{layer}.mlp.c_proj.weight")), vector(weights, f"h.{layer}.mlp.c_proj.bias"))


def logits(weights, config, ids):
    hidden = embed(weights, ids)
    for layer in range(config["n_layer"]):
        hidden = block(weights, config, layer, hidden)
    last = layer_norm(weights, hidden[-1], "ln_f")
    wte = matrix(weights, "wte.weight")
    return [dot(last, row) for row in wte]


def generate(weights, config, ids, count, temperature, top_p, random):
    cache = [[[] for _ in range(config["n_head"])] for _ in range(config["n_layer"])]
    hidden = embed(weights, ids)
    for layer in range(config["n_layer"]):
        hidden = block_from(weights, config, layer, hidden, cache)
    out = []
    sequence = list(ids)
    for step in range(count):
        nxt = pick(weights, hidden[-1], temperature, top_p, None if random is None else random[step])
        out.append(nxt)
        sequence.append(nxt)
        wte = matrix(weights, "wte.weight")
        wpe = matrix(weights, "wpe.weight")
        pos = len(sequence) - 1
        row = [[wte[nxt][j] + wpe[pos][j] for j in range(len(wte[nxt]))]]
        for layer in range(config["n_layer"]):
            row = block_from(weights, config, layer, row, cache)
        hidden = row
    return out


def pick(weights, last, temperature, top_p, random):
    scored = layer_norm(weights, last, "ln_f")
    wte = matrix(weights, "wte.weight")
    if temperature == 0:
        best = 0
        best_score = None
        for i, row in enumerate(wte):
            score = dot(scored, row)
            if best_score is None or score > best_score:
                best = i
                best_score = score
        return best
    logits = [dot(scored, row) / temperature for row in wte]
    return sample_token(logits, top_p, random)


def sample_token(logits, top_p, random):
    probs = softmax(logits)
    order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
    kept = []
    mass = 0.0
    for token in order:
        kept.append(token)
        mass += probs[token]
        if mass >= top_p:
            break
    kept_mass = sum(probs[token] for token in kept)
    cum = 0.0
    for token in kept:
        cum += probs[token] / kept_mass
        if cum > random:
            return token
    return kept[-1]


def gelu(value):
    return 0.5 * value * (1 + math.tanh(math.sqrt(2 / math.pi) * (value + 0.044715 * value * value * value)))


def layer_norm_rows(weights, rows, prefix):
    return [layer_norm(weights, row, prefix) for row in rows]


def layer_norm(weights, row, prefix):
    weight = vector(weights, prefix + ".weight")
    bias = vector(weights, prefix + ".bias")
    n = len(row)
    mean = sum(row) / n
    variance = sum((value - mean) ** 2 for value in row) / n
    inv = 1 / math.sqrt(variance + LN_EPS)
    return [(value - mean) * inv * weight[i] + bias[i] for i, value in enumerate(row)]


def matmul(a, b):
    out = []
    for row in a:
        acc = [0.0] * len(b[0])
        for k, value in enumerate(row):
            bk = b[k]
            for j in range(len(acc)):
                acc[j] += value * bk[j]
        out.append(acc)
    return out


def add_bias(rows, bias):
    return [[value + bias[j] for j, value in enumerate(row)] for row in rows]


def add_rows(a, b):
    return [[a[i][j] + b[i][j] for j in range(len(a[i]))] for i in range(len(a))]


def softmax(values):
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


if __name__ == "__main__":
    main()
