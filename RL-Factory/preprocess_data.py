# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the nq dataset to parquet format
"""

import re
import os
import datasets
import argparse


if __name__ == '__main__':
    train_dataset = datasets.load_dataset('json', data_files=os.environ['HOME'] + '/agentic-nl2sql/data/train.json')['train']
    test_dataset = datasets.load_dataset('json', data_files=os.environ['HOME'] + '/agentic-nl2sql/data/val.json')['train']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            data = {
                "prompt": example['prompt'],
                "ability": "nl2sql",
                "reward_model": {
                    "style": "rule",
                    "db_id": example['db_id'],
                    "ground_truth": example['gold_sql']
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    train_dataset.to_parquet(os.environ['HOME'] + '/agentic-nl2sql/RL-Factory/data/train.parquet')
    test_dataset.to_parquet(os.environ['HOME'] + '/agentic-nl2sql/RL-Factory/data/test.parquet')

    train_dataset.to_json(
        os.environ['HOME'] + '/agentic-nl2sql/RL-Factory/data/train.json',
        orient='records',
        force_ascii=False,
        indent=4
    )
    test_dataset.to_json(
        os.environ['HOME'] + '/agentic-nl2sql/RL-Factory/data/test.json',
        orient='records',
        force_ascii=False,
        indent=4
    )