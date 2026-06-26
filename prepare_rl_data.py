import asyncio
import json
import os
import re
import transformers
from openai import AsyncOpenAI
from tqdm import tqdm

from utils import load_json, exec_sql, sync_compare_sql

problems = load_json("./bird/train.json")
schemas = load_json("./bird/schemas.json")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    os.path.expanduser("~/model/agent-7b")
)

client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:10080/v1",
)
model_name = "qwen1"

MAX_CONTEXT_LENGTH = 8192
MAX_NEW_TOKENS = 1024

curr_search_template = '\n{output_text}\n<sql_exec_result>\n{search_results}\n</sql_exec_result>\n'

prompt_template = """You are a SQL expert. Given the [Question], [DB Schema], and [Hints], you are tasked to transform the [Question] into an executable SQLite query. \
You must conduct reasoning inside <think> and </think> first, and put the transformed SQL into <sql> and </sql>. \
You can get the execution feedback of your SQL between <sql_exec_result> and </sql_exec_result>. \
If there are any grammar error or the execution result is empty, you need to re-conduct reasoning, and rewrite the refined SQL between <sql> and </sql>. \
If the SQL query exeuctes successfully with concrete results, you can briefly summarize your solution between <think> and </think>, and provide the final SQL between <final_sql> and </final_sql>.
Question: {question}
DB Schema: {db_schema}
Hints: {hints}
"""

N_ATTEMPTS = 20


def get_sql(text):
    pattern = re.compile(r"<sql>(.*?)</sql>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    return None


def get_final_sql(text):
    pattern = re.compile(r"<final_sql>(.*?)</final_sql>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    return None


async def process_problem(idx, prob, sem, file_lock, pbar):
    async with sem:
        question = prob['question']
        db_id = prob['db_id']
        db_schema = schemas[db_id]
        hints = prob['evidence']
        gold_sql = prob['SQL']

        prompt_text = prompt_template.format(question=question, db_id=db_id, db_schema=db_schema, hints=hints)
        attempts = []

        for _ in range(N_ATTEMPTS):
            prompt = prompt_text
            trajectory = [prompt.strip()]

            while True:
                prompt_token_count = len(tokenizer.encode(prompt))
                if prompt_token_count + MAX_NEW_TOKENS > MAX_CONTEXT_LENGTH:
                    break

                try:
                    response = await client.completions.create(
                        model=model_name,
                        prompt=prompt,
                        max_tokens=MAX_NEW_TOKENS,
                        temperature=1.0,
                        stop=["</sql>"],
                        timeout=120,
                    )
                except Exception as e:
                    print(f"\n[Error] vLLM API Error on Problem {idx}: {e}")
                    break

                output_text = response.choices[0].text
                think_start = output_text.find('<think>')
                if think_start != -1:
                    output_text = output_text[think_start:]
                finish_reason = response.choices[0].finish_reason

                is_final = False
                if finish_reason == "stop":
                    if "<final_sql>" in output_text:
                        is_final = True
                    else:
                        output_text += "</sql>"

                if is_final or finish_reason == "length":
                    trajectory.append(output_text.strip())
                    break

                tmp_query = get_sql(prompt + output_text)
                if tmp_query:
                    search_results = await asyncio.to_thread(exec_sql, tmp_query, db_id)
                else:
                    search_results = ''

                search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
                prompt += search_text
                trajectory.append(search_text.strip())

            pred_sql = get_final_sql(trajectory[-1]) if trajectory else None
            if pred_sql:
                eval_status, eval_res = await asyncio.to_thread(sync_compare_sql, pred_sql, gold_sql, db_id)
                attempts.append({
                    'trajectory': trajectory,
                    'pred_sql': pred_sql,
                    'eval_res': eval_res,
                })

        corrects = [a for a in attempts if a.get('eval_res') == '正确']
        wrongs = [a for a in attempts if a.get('eval_res') != '正确']
        if corrects and wrongs:
            record = {
                'db_id': db_id,
                'question': question,
                'evidence': hints,
                'SQL': gold_sql,
                'n_correct': len(corrects),
                'n_wrong': len(wrongs),
            }
            async with file_lock:
                with open('./bird/rl_data.json', 'a') as f:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

        pbar.update(1)



async def main():
    concurrency_limit = 20
    sem = asyncio.Semaphore(concurrency_limit)
    file_lock = asyncio.Lock()
    pbar = tqdm(total=len(problems), desc="Preparing RL data")

    tasks = []
    for idx, prob in enumerate(problems[:]):
        task = asyncio.create_task(process_problem(idx, prob, sem, file_lock, pbar))
        tasks.append(task)

    await asyncio.gather(*tasks)
    pbar.close()


if __name__ == "__main__":
    asyncio.run(main())
