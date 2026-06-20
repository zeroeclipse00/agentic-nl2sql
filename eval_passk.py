"""
Pass@k baseline evaluation for the SFT model.

Usage:
  1. Start vLLM server first:  bash start_vllm_sft.sh
  2. Run this script:          python eval_passk.py [--split train|dev] [--n 8] [--limit 500]

Metrics reported: pass@1, pass@4, pass@8 (unbiased estimator)
"""

import asyncio
import argparse
import json
import math
from math import comb
import httpx
from openai import AsyncOpenAI
import transformers
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_json, load_jsonl, sync_compare_sql, exec_sql


# --- Config ---
VLLM_PORT    = 10082
MODEL_NAME   = "sft-agent-7b"
TOKENIZER_ID = "/var/tmp/wangshangshu/models/agent-7b-hf"
MAX_CONTEXT  = 8192
MAX_NEW_TOKENS = 1024

# Disable system proxy — vLLM is on localhost, proxy would break the connection
client = AsyncOpenAI(
    api_key="EMPTY",
    base_url=f"http://localhost:{VLLM_PORT}/v1",
    http_client=httpx.AsyncClient(trust_env=False),
)
tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_ID)

SEARCH_TMPL = '\n{output_text}\n<sql_exec_result>\n{search_results}\n</sql_exec_result>\n'

PROMPT_TMPL = """You are a SQL expert. Given the [Question], [DB Schema], and [Hints], \
you are tasked to transform the [Question] into an executable SQLite query. \
You must conduct reasoning inside <think> and </think> first, and put the transformed SQL into <sql> and </sql>. \
You can get the execution feedback of your SQL between <sql_exec_result> and </sql_exec_result>. \
If there are any grammar error or the execution result is empty, you need to re-conduct reasoning, \
and rewrite the refined SQL between <sql> and </sql>. \
If the SQL query executes successfully with concrete results, you can briefly summarize your solution \
between <think> and </think>, and provide the final SQL between <final_sql> and </final_sql>.
Question: {question}
DB Schema: {db_schema}
Hints: {hints}
"""


def get_sql(text):
    import re
    matches = re.compile(r"<sql>(.*?)</sql>", re.DOTALL).findall(text)
    return matches[-1].strip() if matches else None


def get_final_sql(text):
    import re
    matches = re.compile(r"<final_sql>(.*?)</final_sql>", re.DOTALL).findall(text)
    return matches[-1].strip() if matches else None


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator: probability that at least one of k samples is correct."""
    if n == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


async def run_one_attempt(prompt: str, db_id: str) -> tuple[str | None, list[str]]:
    """Run one agentic attempt, return (pred_sql, trajectory)."""
    trajectory = [prompt.strip()]
    cur_prompt = prompt

    for _ in range(5):  # max 5 turns
        if len(tokenizer.encode(cur_prompt)) + MAX_NEW_TOKENS > MAX_CONTEXT:
            break
        try:
            resp = await client.completions.create(
                model=MODEL_NAME,
                prompt=cur_prompt,
                max_tokens=MAX_NEW_TOKENS,
                temperature=1.0,
                stop=["</sql>"],
            )
        except Exception as e:
            break

        output = resp.choices[0].text
        reason = resp.choices[0].finish_reason

        is_final = reason == "stop" and "<final_sql>" in output
        if is_final or reason == "length":
            trajectory.append(output.strip())
            break

        tmp_sql = get_sql(cur_prompt + output)
        search = await asyncio.to_thread(exec_sql, tmp_sql, db_id) if tmp_sql else ''
        search_text = SEARCH_TMPL.format(output_text=output, search_results=search)
        cur_prompt += search_text
        trajectory.append(search_text.strip())

    pred_sql = get_final_sql(trajectory[-1]) if trajectory else None
    return pred_sql, trajectory


async def process_problem(idx, prob, schemas, n_samples, sem, file_lock, pbar, out_file):
    async with sem:
        question  = prob['question']
        db_id     = prob['db_id']
        db_schema = schemas[db_id]
        hints     = prob.get('evidence', '')
        gold_sql  = prob['SQL']

        base_prompt = PROMPT_TMPL.format(
            question=question, db_id=db_id, db_schema=db_schema, hints=hints
        )

        attempts = []
        tasks = [run_one_attempt(base_prompt, db_id) for _ in range(n_samples)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for pred_sql, traj in results:
            if isinstance(pred_sql, Exception):
                continue
            correct = False
            if pred_sql:
                _, cmp = await asyncio.to_thread(sync_compare_sql, pred_sql, gold_sql, db_id)
                correct = (cmp == '正确')
            attempts.append({'pred_sql': pred_sql, 'correct': correct})

        n = len(attempts)
        c = sum(1 for a in attempts if a['correct'])

        record = {
            'idx': idx,
            'question': question,
            'db_id': db_id,
            'gold_sql': gold_sql,
            'n_samples': n,
            'n_correct': c,
            'pass@1':  pass_at_k(n, c, 1),
            'pass@5':  pass_at_k(n, c, 5),
            'pass@10': pass_at_k(n, c, 10),
            'attempts': attempts,
        }

        async with file_lock:
            with open(out_file, 'a') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        pbar.update(1)
        return record


async def main(args):
    if args.split == 'dev':
        problems = load_json('./bird/dev.json')
    else:
        problems = load_json('./bird/train.json')
    schemas = load_json('./bird/schemas.json')

    if args.limit:
        problems = problems[:args.limit]

    out_file = f"passk_{args.split}_n{args.n}.jsonl"
    # Clear previous results
    open(out_file, 'w').close()

    sem = asyncio.Semaphore(args.concurrency)
    file_lock = asyncio.Lock()
    pbar = tqdm(total=len(problems), desc=f"Pass@k eval ({args.split})")

    tasks = [
        asyncio.create_task(
            process_problem(idx, prob, schemas, args.n, sem, file_lock, pbar, out_file)
        )
        for idx, prob in enumerate(problems)
    ]
    records = await asyncio.gather(*tasks)
    pbar.close()

    # Aggregate metrics
    valid = [r for r in records if not isinstance(r, Exception) and r is not None]
    n_total = len(valid)
    if n_total == 0:
        print("No valid records.")
        return

    avg_p1  = sum(r['pass@1']  for r in valid) / n_total
    avg_p5  = sum(r['pass@5']  for r in valid) / n_total
    avg_p10 = sum(r['pass@10'] for r in valid) / n_total

    print(f"\n{'='*50}")
    print(f"SFT Model Baseline  ({args.split}, n={args.n}, N={n_total})")
    print(f"  pass@1  : {avg_p1:.4f}  ({avg_p1*100:.2f}%)")
    print(f"  pass@5  : {avg_p5:.4f}  ({avg_p5*100:.2f}%)")
    print(f"  pass@10 : {avg_p10:.4f}  ({avg_p10*100:.2f}%)")
    print(f"Results saved to: {out_file}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",       default="dev",  choices=["train", "dev"])
    parser.add_argument("--n",           type=int, default=10, help="samples per problem")
    parser.add_argument("--limit",       type=int, default=None, help="limit number of problems (None=all)")
    parser.add_argument("--concurrency", type=int, default=8,  help="async concurrency")
    args = parser.parse_args()
    asyncio.run(main(args))
