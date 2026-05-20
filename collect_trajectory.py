import asyncio
from openai import AsyncOpenAI  # 引入异步客户端
from utils import load_json, load_jsonl, save_json, sync_exec_sql, sync_compare_sql, exec_sql
import transformers
from tqdm import tqdm
import json

problems = load_json('./bird/train.json')
schemas = load_json('./bird/schemas.json')

# 1. 本地分词器（仅用于统计 Token 数量）
tokenizer_id = "/home/koujianshang/models/Qwen2.5-Coder-32B-Instruct"
tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id)

# 2. 异步 vLLM 客户端设置
client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:10091/v1",
)
model_name = "qwen"

# 上下文参数控制
MAX_CONTEXT_LENGTH = 8192
MAX_NEW_TOKENS = 1024

curr_search_template = '\n{output_text}\n<sql_exec_result>\n{search_results}\n</sql_exec_result>\n'

prompt_template = """You are a SQL expert. Given the [Question], [DB Schema], and [Hints], you are tasked to transform the [Question] into an executable SQLite query. \
You must conduct reasoning inside <reason> and </reason> first, and put the transformed SQL into <sql> and </sql>. \
You can get the execution feedback of your SQL between <sql_exec_result> and </sql_exec_result>. \
If there are any grammar error or the execution result is empty, you need to re-conduct reasoning, and rewrite the refined SQL between <sql> and </sql>. \
If the SQL query exeuctes successfully with concrete results, you can briefly summarize your solution between <reason> and </reason>, and provide the final SQL between <final_sql> and </final_sql>.
Question: {question}
DB Schema: {db_schema}
Hints: {hints}
"""

def get_sql(text):
    import re
    pattern = re.compile(r"<sql>(.*?)</sql>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    return None

def get_final_sql(text):
    import re
    pattern = re.compile(r"<final_sql>(.*?)</final_sql>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    return None

N_ATTEMPTS = 5

# 核心异步处理函数
async def process_problem(idx, prob, sem, file_lock, pbar):
    async with sem:  # 限制并发数为16
        question = prob['question']
        db_id = prob['db_id']
        db_schema = schemas[db_id]
        hints = prob['evidence']
        gold_sql = prob['SQL']
        initial_prompt = prompt_template.format(question=question, db_id=db_id, db_schema=db_schema, hints=hints)
        meta_info = {
            'id': idx,
            'question': question,
            'db_id': db_id, 
            'hints': hints,
            'gold_sql': gold_sql,
            'prompt': initial_prompt
        }
        attempts = []
        
        for i in range(N_ATTEMPTS):
            prompt = initial_prompt
            trajectory = []
            
            while True:
                prompt_token_count = len(tokenizer.encode(prompt))
                if prompt_token_count + MAX_NEW_TOKENS > MAX_CONTEXT_LENGTH:
                    # 避免控制台输出混乱，异步模式下建议用 logging，这里保留 print 但需注意可能会穿插
                    # print(f"\n[Warning] Problem {idx} prompt exceeded max context length.")
                    break 
                    
                try:
                    # 发送异步 API 请求
                    response = await client.completions.create(
                        model=model_name,
                        prompt=prompt,
                        max_tokens=MAX_NEW_TOKENS,
                        temperature=1.0,
                        stop=["</sql>"],
                        extra_body={"chat_template_kwargs":{"enable_thinking":False}}
                    )
                except Exception as e:
                    print(f"\n[Error] vLLM API Error on Problem {idx}: {e}")
                    break

                output_text = response.choices[0].text
                reason_start = output_text.find('<reason>')
                output_text = output_text[reason_start:]
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
                    # 将同步的数据库执行操作放到线程池中运行，避免阻塞 event loop
                    search_results = await asyncio.to_thread(exec_sql, tmp_query, db_id)
                else:
                    search_results = ''
            
                search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
                prompt += search_text
                trajectory.append(search_text.strip())

            pred_sql = get_final_sql(trajectory[-1]) if trajectory else None
            if pred_sql:
                # 同理，将同步对比 SQL 的操作放入线程池
                eval_status, eval_res = await asyncio.to_thread(sync_compare_sql, pred_sql, gold_sql, db_id)
                if eval_res == '正确':
                    attempts.append({
                        'trajectory': trajectory,
                        'pred_sql': pred_sql,
                        'eval_res': eval_res
                    })
                    
        meta_info['attempts'] = attempts
        if attempts:
            # 异步写文件，必须加锁防止多协程同时写入导致行断裂
            async with file_lock:
                # 注意：这里使用内置的 open，在加锁的保护下且写入量不大时是安全的。
                # 如果追求极致性能，可考虑安装使用 aiofiles 库。
                with open('qwen2.5-coder-32b.0520.jsonl', 'a') as f:
                    f.write(json.dumps(meta_info, ensure_ascii=False) + '\n')
        
        # 完成一个任务，更新进度条
        pbar.update(1)

# 调度器主函数
async def main():
    concurrency_limit = 16
    sem = asyncio.Semaphore(concurrency_limit)
    file_lock = asyncio.Lock()
    
    # 初始化进度条
    pbar = tqdm(total=len(problems), desc="Processing Problems")
    
    # 创建所有任务
    tasks = []
    for idx, prob in enumerate(problems[:]):
        task = asyncio.create_task(process_problem(idx, prob, sem, file_lock, pbar))
        tasks.append(task)
        
    # 等待所有任务执行完毕
    await asyncio.gather(*tasks)
    pbar.close()

if __name__ == "__main__":
    # 启动异步事件循环
    asyncio.run(main())
