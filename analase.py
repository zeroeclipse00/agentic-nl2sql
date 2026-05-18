#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import sys
from pathlib import Path

def jsonl_to_json(input_path: str):
    # 自动生成同名json路径
    input_file = Path(input_path)
    output_file = input_file.with_suffix('.json')
    
    json_data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            json_data.append(item)
    return json_data

data = jsonl_to_json(sys.argv[1])

total = len(data)

pass_10 = 0

for item in data:
    ress = [attempt['eval_res'] for attempt in item['attempts']][:]
    pass_10 += '正确' in ress

print(f'Pass@10: {pass_10} / {total} = {pass_10 / total: .2f}')
    

