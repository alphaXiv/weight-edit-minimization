"""Build a candidate checkpoint: base Qwen3.5-0.8B snapshot + a subset of Astra's published sparse edit
(exp8_math/runs/p50c/edit/edit.npz, bf16 exact) applied to the two down_proj tensors.
Spec JSON: {"cols15": [..], "cols19": [..], "rows15": [..], "rows19": [..],
            "entries15": [[r,c], ..], "entries19": [...], "scale": 1.0}
Cols/rows are absolute neuron indices into the 3584 / 1024 dims. Missing key = keep everything (subject to other filters).
scale multiplies each stored delta (recomputed in float32, re-rounded to bf16) — only use 1.0 for bit-exact subsets."""
import argparse, json, os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import st_numpy as S

TENSORS = [
    'model.language_model.layers.15.mlp.down_proj.weight',
    'model.language_model.layers.19.mlp.down_proj.weight',
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--spec', required=True)
    ap.add_argument('--edit', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'exp8', 'edit.npz'))
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    SNAP = snapshot_download('Qwen/Qwen3.5-0.8B', revision='2fc06364715b967f1860aea9cf38778875588b17', token=False)
    spec = json.load(open(a.spec))
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(SNAP):
        if f == 'model.safetensors-00001-of-00001.safetensors':
            continue
        dst = os.path.join(a.out, f)
        if not os.path.exists(dst):
            os.symlink(os.path.join(SNAP, f), dst)
    z = np.load(a.edit)
    base = S.load(glob.glob(SNAP + '/*.safetensors')[0], TENSORS)
    out = {}
    for i, name in enumerate(TENSORS):
        key = '15' if i == 0 else '19'
        dt, bits = base[name]
        idx = z[f't{i}_idx'].astype(np.int64)
        val = z[f't{i}_val']
        shape = bits.shape
        flat = bits.reshape(-1).copy()
        rows = idx // shape[1]
        cols = idx % shape[1]
        keep = np.ones(len(idx), dtype=bool)
        if f'cols{key}' in spec:
            cs = set(spec[f'cols{key}']); keep &= np.isin(cols, list(cs))
        if f'rows{key}' in spec:
            rs = set(spec[f'rows{key}']); keep &= np.isin(rows, list(rs))
        if f'entries{key}' in spec:
            es = set(map(tuple, spec[f'entries{key}']))
            keep &= np.array([(int(r), int(c)) in es for r, c in zip(rows, cols)])
        if spec.get('scale', 1.0) != 1.0:
            scale = spec['scale']
            cur = S.bf16_to_f32(val[keep]).astype(np.float64)          # stored (edited) bits
            orig = S.bf16_to_f32(base[name][1].reshape(-1)[idx[keep]]).astype(np.float64)
            delta = (cur - orig) * scale
            flat[idx[keep]] = S.bf16_bits((orig + delta).astype(np.float32))
        else:
            flat[idx[keep]] = val[keep]
        out[name] = ('BF16', flat.reshape(shape))
    full = S.load(glob.glob(SNAP + '/*.safetensors')[0])
    for n in list(full):
        if n in out:
            full[n] = out[n]
    S.save(os.path.join(a.out, 'model.safetensors-00001-of-00001.safetensors'), full, {'format': 'pt'})
    print('wrote', a.out)

if __name__ == '__main__':
    main()
