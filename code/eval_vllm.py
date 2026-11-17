"""Minimal vLLM eval harness replicating the Experiment 8 protocol:
chat template with enable_thinking=True, user msg = problem + instruction, think35 sampling
(temp 0.6, top_p 0.95, top_k 20, presence_penalty 1.5, max 32768 tokens, seed 0), last \boxed{} graded
with math-verify. Prints RESULT lines and writes per-problem JSONL."""
import argparse, json, os, time
import numpy as np

INSTR = "Please reason step by step, and put your final answer within \\boxed{}."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--problems', required=True, help='jsonl with id/problem/answer')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=1)
    ap.add_argument('--max-tokens', type=int, default=32768)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--gpu-mem', type=float, default=0.45)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if os.path.exists(a.out):  # fail before the hours-long run, not after it
        ap.error(f'{a.out} already exists; choose another --out')

    rows = [json.loads(l) for l in open(a.problems)]
    rows = rows[a.start:a.start + a.limit] if a.limit else rows[a.start:]

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, dtype='bfloat16', gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_tokens + 1024, seed=a.seed, enforce_eager=False)
    prompts = [tok.apply_chat_template([{'role': 'user', 'content': r['problem'] + '\n' + INSTR}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=True) for r in rows]
    sp = SamplingParams(n=a.n, temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5,
                        max_tokens=a.max_tokens, seed=a.seed)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    wall = time.time() - t0

    from math_verify import parse, verify
    tot = ok = cut = 0
    toks = []
    with open(a.out, 'x') as f:  # 'x': a run started meanwhile with the same --out keeps its result
        for r, o in zip(rows, outs):
            cs = []
            for s in o.outputs:
                try:
                    good = bool(verify(parse(f"${r['answer']}$"), parse(s.text)))
                except Exception:
                    good = False
                cs.append(int(good))
                cut += 1 if s.finish_reason == 'length' else 0
                toks.append(len(s.token_ids))
                tot += 1
                ok += int(good)
            f.write(json.dumps({'id': r['id'], 'correct': cs, 'finish': [s.finish_reason for s in o.outputs],
                                'n_tokens': [len(s.token_ids) for s in o.outputs]}) + '\n')
    print('RESULT', json.dumps({'model': a.model, 'n_problems': len(rows), 'samples': a.n,
                                'acc': ok / max(tot, 1), 'cut_share': cut / max(tot, 1),
                                'mean_tokens': float(np.mean(toks)) if toks else 0,
                                'wall_s': round(wall, 1), 'out': a.out}))

if __name__ == '__main__':
    main()
