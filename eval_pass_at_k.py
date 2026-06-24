"""Evaluate pass@k on BIRD-dev using vLLM for generation and sql_server for execution."""
import os
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']:
    os.environ.pop(k, None)

import json
import re
import sys
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from openai import OpenAI
from utils import load_json, sync_compare_sql

schemas = load_json('./bird/schemas.json')
dev_data = load_json('./bird/dev.json')

prompt_template = """You are a SQL expert. Given the [Question], [DB Schema], and [Hints], you are tasked to transform the [Question] into an executable SQLite query. \
You must conduct reasoning inside <think> and </think> first, and put the transformed SQL into <sql> and </sql>. \
You can get the execution feedback of your SQL between <sql_exec_result> and </sql_exec_result>. \
If there are any grammar error or the execution result is empty, you need to re-conduct reasoning, and rewrite the refined SQL between <sql> and </sql>. \
If the SQL query exeuctes successfully with concrete results, you can briefly summarize your solution between <think> and </think>, and provide the final SQL between <final_sql> and </final_sql>.
Question: {question}
DB Schema: {db_schema}
Hints: {hints}
"""


def extract_final_sql(text):
    matches = list(re.finditer(r'<final_sql>(.*?)</final_sql>', text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    matches = list(re.finditer(r'<sql>(.*?)</sql>', text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None


def evaluate_one(pred_sql, gold_sql, db_id):
    if pred_sql is None:
        return False
    try:
        _, result = sync_compare_sql(pred_sql, gold_sql, db_id)
        return result == '正确'
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--port', type=int, default=10086)
    parser.add_argument('--max-tokens', type=int, default=1024)
    args = parser.parse_args()

    client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{args.port}/v1")

    results = []
    pass_counts = defaultdict(int)

    for item in tqdm(dev_data, desc="Evaluating"):
        db_id = item['db_id']
        gold_sql = item['SQL']
        prompt = prompt_template.format(
            question=item['question'],
            db_schema=schemas.get(db_id, ''),
            hints=item.get('evidence', ''),
        )

        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                n=args.k,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as e:
            print(f"API error for {db_id}: {e}")
            results.append({'db_id': db_id, 'pass': [False] * args.k})
            continue

        sample_results = []
        for choice in response.choices:
            pred_sql = extract_final_sql(choice.message.content)
            correct = evaluate_one(pred_sql, gold_sql, db_id)
            sample_results.append(correct)

        results.append({
            'db_id': db_id,
            'gold_sql': gold_sql,
            'pass': sample_results,
        })

    # Compute pass@k
    total = len(results)
    for k in range(1, args.k + 1):
        pass_at_k = sum(1 for r in results if any(r['pass'][:k])) / total
        print(f"pass@{k} = {pass_at_k:.4f} ({sum(1 for r in results if any(r['pass'][:k]))}/{total})")

    # Per-difficulty breakdown
    diff_map = {item['question_id']: item.get('difficulty', 'unknown') for item in dev_data}
    for diff in ['simple', 'moderate', 'challenging']:
        subset = [r for r, item in zip(results, dev_data) if item.get('difficulty') == diff]
        if subset:
            for k in range(1, args.k + 1):
                pass_at_k = sum(1 for r in subset if any(r['pass'][:k])) / len(subset)
                print(f"  {diff} pass@{k} = {pass_at_k:.4f} ({sum(1 for r in subset if any(r['pass'][:k]))}/{len(subset)})")

    # Save results
    out_path = f"eval_results_pass{args.k}.json"
    with open(out_path, 'w') as f:
        json.dump({'results': results, 'model': args.model, 'k': args.k}, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
