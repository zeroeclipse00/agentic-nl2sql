import random
from utils import load_jsonl, save_json

data = load_jsonl('./trajectories/trajectory.32b-inst.jsonl')[:]

postprocessed_data = []

for item in data:
    attempts = item['attempts']
    prompt = item['prompt']
    for attempt in attempts:
        traj = attempt['trajectory']
        answers = []
        for think_and_sql in traj:
            # 偶尔出现 <think> 标签前也有思考的情况，清理一下
            cleaned_think_and_sql = think_and_sql[think_and_sql.find('<think>'):]
            answers.append(cleaned_think_and_sql)
        answer = '\n'.join(answers)
        postprocessed_data.append({
            'db_id': item['db_id'],
            'prompt': prompt,
            'answer': answer,
            'gold_sql': item['gold_sql']
        })

random.shuffle(postprocessed_data)

save_json(postprocessed_data[:int(0.8 * len(postprocessed_data))], './data/train.json')
save_json(postprocessed_data[int(0.8 * len(postprocessed_data)):], './data/val.json')
