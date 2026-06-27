"""
Pass@k agentic evaluation for NL2SQL models.

Usage:
  1. Start vLLM server:   vllm serve <model> --port 10086 --served-model-name sft-agent-7b ...
  2. Start SQL server:    python bird/sql_server.py --port 11111
  3. Run eval:            python eval_passk.py [--n 1] [--temperature 0] [--max-turns 3]

Metrics reported: pass@1 (and pass@k for n > 1, unbiased estimator)
"""

import asyncio
import argparse
import json
from math import comb
import httpx
from openai import AsyncOpenAI
import transformers
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_json, load_jsonl, sync_compare_sql, exec_sql


MAX_CONTEXT    = 8192
MAX_NEW_TOKENS = 1024

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


async def run_one_attempt(
    prompt: str, db_id: str, client: AsyncOpenAI, tokenizer,
    model: str, temperature: float, max_turns: int
) -> tuple[str | None, str]:
    """Run one agentic attempt (up to max_turns SQL executions).

    Returns (pred_sql, full_context).
    Bugs fixed vs original:
      - Apply chat template so the model sees the correct input format
      - Add </sql> back after stopping at that token (otherwise get_sql fails)
      - Extract pred_sql from full accumulated context, not just last trajectory chunk
    """
    cur_prompt = prompt

    for _ in range(max_turns):
        if len(tokenizer.encode(cur_prompt)) + MAX_NEW_TOKENS > MAX_CONTEXT:
            break
        try:
            resp = await client.completions.create(
                model=model,
                prompt=cur_prompt,
                max_tokens=MAX_NEW_TOKENS,
                temperature=temperature,
                stop=["</sql>"],
            )
        except Exception:
            break

        output = resp.choices[0].text
        reason = resp.choices[0].finish_reason

        # Model gave final answer (stopped at EOS with <final_sql>) or hit token limit
        if (reason == "stop" and "<final_sql>" in output) or reason == "length":
            cur_prompt += output
            break

        # Model made a SQL call — stopped at </sql>, so add the closing tag back
        output_with_close = output + "</sql>"
        cur_prompt += output_with_close

        tmp_sql = get_sql(cur_prompt)
        sql_result = await asyncio.to_thread(exec_sql, tmp_sql, db_id) if tmp_sql else "No SQL found"
        cur_prompt += f"<sql_exec_result>\n{sql_result}\n</sql_exec_result>\n"

    # Extract answer from full context; fall back to last <sql> if no <final_sql>
    pred_sql = get_final_sql(cur_prompt) or get_sql(cur_prompt)
    return pred_sql, cur_prompt


async def process_problem(idx, prob, schemas, n_samples, sem, file_lock, pbar, out_file,
                          client, tokenizer, model, temperature, max_turns):
    async with sem:
        question  = prob['question']
        db_id     = prob['db_id']
        db_schema = schemas[db_id]
        hints     = prob.get('evidence', '')
        gold_sql  = prob['SQL']

        # Apply chat template so the model sees the same format as during training
        raw_content = PROMPT_TMPL.format(
            question=question, db_schema=db_schema, hints=hints
        )
        base_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_content}],
            tokenize=False,
            add_generation_prompt=True,
        )

        attempts = []
        tasks = [
            run_one_attempt(base_prompt, db_id, client, tokenizer, model, temperature, max_turns)
            for _ in range(n_samples)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                continue
            pred_sql, _ = result
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
            'pass@1': pass_at_k(n, c, 1),
            'attempts': attempts,
        }
        if n >= 5:
            record['pass@5'] = pass_at_k(n, c, 5)
        if n >= 10:
            record['pass@10'] = pass_at_k(n, c, 10)

        async with file_lock:
            with open(out_file, 'a') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        pbar.update(1)
        return record


async def main(args):
    # Build shared client and tokenizer
    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{args.port}/v1",
        http_client=httpx.AsyncClient(trust_env=False),
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

    if args.split == 'dev':
        problems = load_json('./bird/dev.json')
    else:
        problems = load_json('./bird/train.json')
    schemas = load_json('./bird/schemas.json')

    if args.limit:
        problems = problems[:args.limit]

    out_file = f"passk_{args.split}_n{args.n}_t{args.temperature}.jsonl"
    open(out_file, 'w').close()

    sem = asyncio.Semaphore(args.concurrency)
    file_lock = asyncio.Lock()
    pbar = tqdm(total=len(problems), desc=f"Pass@k eval ({args.split}, n={args.n}, T={args.temperature})")

    tasks = [
        asyncio.create_task(
            process_problem(
                idx, prob, schemas, args.n, sem, file_lock, pbar, out_file,
                client, tokenizer, args.model, args.temperature, args.max_turns,
            )
        )
        for idx, prob in enumerate(problems)
    ]
    records = await asyncio.gather(*tasks)
    pbar.close()

    valid = [r for r in records if not isinstance(r, Exception) and r is not None]
    n_total = len(valid)
    if n_total == 0:
        print("No valid records.")
        return

    avg_p1 = sum(r['pass@1'] for r in valid) / n_total
    print(f"\n{'='*50}")
    print(f"Model: {args.model}  split={args.split}  n={args.n}  T={args.temperature}  max_turns={args.max_turns}  N={n_total}")
    print(f"  pass@1  : {avg_p1:.4f}  ({avg_p1*100:.2f}%)")
    if args.n >= 5:
        avg_p5 = sum(r.get('pass@5', 0) for r in valid) / n_total
        print(f"  pass@5  : {avg_p5:.4f}  ({avg_p5*100:.2f}%)")
    if args.n >= 10:
        avg_p10 = sum(r.get('pass@10', 0) for r in valid) / n_total
        print(f"  pass@10 : {avg_p10:.4f}  ({avg_p10*100:.2f}%)")
    print(f"Results saved to: {out_file}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",       default="dev", choices=["train", "dev"])
    parser.add_argument("--n",           type=int,   default=1,    help="samples per problem")
    parser.add_argument("--temperature", type=float, default=0.0,  help="sampling temperature (0=greedy)")
    parser.add_argument("--max-turns",   type=int,   default=3,    help="max agentic turns (match training)")
    parser.add_argument("--port",        type=int,   default=10086, help="vLLM server port")
    parser.add_argument("--model",       type=str,   default="sft-agent-7b", help="served model name")
    parser.add_argument("--tokenizer",   type=str,   default="/root/autodl-tmp/models/agent-7b-6656")
    parser.add_argument("--limit",       type=int,   default=None, help="limit problems (None=all)")
    parser.add_argument("--concurrency", type=int,   default=8,    help="async concurrency")
    args = parser.parse_args()
    asyncio.run(main(args))
