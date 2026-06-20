import re
import json
import string
import random
import torch
from .base import Env

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, "../.."))


if project_root not in sys.path:
    sys.path.append(project_root)

from utils import sync_compare_sql, sync_exec_sql

class NL2SQLEnv(Env):
    def __init__(self, config, centralized_actor=None):
        super().__init__(config, centralized_actor)
        self.use_verify_tool = False

    # NOTE: Add your reward calculation rules here!
    def _compute_score_with_rules(self, data, tokenizer, if_val=False):

        def em_check(prediction, golden_answer, db_id):
            status, result = sync_compare_sql(prediction, golden_answer, db_id)
            return result == '正确'

        def extract_solution(solution_str):
            """Extract the equation from the solution string."""
            think_pattern = r'<think>.*?</think>'
            solution_str = re.sub(think_pattern, '', solution_str, flags=re.DOTALL)

            answer_pattern = r'<final_sql>(.*?)</final_sql>'
            match = re.finditer(answer_pattern, solution_str, re.DOTALL)
            matches = list(match)
            if len(matches) <= 0:
                return None
            
            return matches[-1].group(1).strip()

        def compute_score_em(solution_str, ground_truth, db_id, format_score=0.0, score=1.):
            """The scoring function for exact match (EM).

            Args:
                solution_str: the solution text
                ground_truth: the ground truth
                format_score: the score for format correctness
                score: the score for a correct answer
            """
            answer = extract_solution(solution_str=solution_str)
            do_print = random.randint(1, 64) == 1

            if do_print:
                print(f"--------------------------------")
                print(f"Golden answers: {ground_truth}")
                print(f"Extracted answer: {answer}")

            # --- Format score: proper alternating <answer> tags ---
            answer_format_score = format_score if check_alternate_tags(solution_str, r"</?answer>") else (-1 * format_score)

            # --- Count SQL tool calls to measure self-correction efficiency ---
            sql_calls = re.findall(r'"name"\s*:\s*"sql_mcp"', solution_str)
            num_calls = len(sql_calls)

            # --- No final SQL extracted: model failed to answer ---
            if answer is None:
                return -format_score + 0.5 * answer_format_score

            # --- Check correctness via sql_server ---
            cmp_status, cmp_result = sync_compare_sql(answer, ground_truth, db_id)

            if cmp_result == '正确':
                # Efficiency bonus: reward fewer self-correction turns
                # 1 call → +0.10, 2 calls → +0.05, 3+ calls → +0.00
                eff_bonus = max(0.0, min(0.10, (3 - num_calls) * 0.05))
                return score + eff_bonus + 0.5 * answer_format_score

            elif cmp_result == '存在多余列':
                # Got correct rows but selected extra columns — partial credit
                return 0.3 + 0.5 * answer_format_score

            elif isinstance(cmp_result, str) and cmp_result.startswith('语法错误'):
                # SQL failed to execute — clear negative signal
                return -0.2 + 0.5 * answer_format_score

            else:
                # SQL executed but result doesn't match ('结果不匹配')
                # Distinguish empty-result trap from semantically wrong SQL
                exec_status, exec_result = sync_exec_sql(answer, db_id)
                if exec_status == 1 and (not exec_result or exec_result == []):
                    # Empty result — common pitfall, penalize more
                    return -0.1 + 0.5 * answer_format_score
                else:
                    # Non-empty but wrong — slight credit for executable SQL
                    return 0.1 + 0.5 * answer_format_score

        def check_alternate_tags(text, tag_pattern):
            # 用正则提取标签名
            match = re.match(r"<\/?(\w+)>", re.findall(tag_pattern, text)[0]) if re.findall(tag_pattern, text) else None
            if not match:
                return False
            tagname = match.group(1)
            open_tag = f"<{tagname}>"
            close_tag = f"</{tagname}>"

            tags = re.findall(tag_pattern, text)

            stack = []
            for tag in tags:
                if tag == open_tag:
                    if stack:
                        # 发现有嵌套，说明不是严格交替
                        return False
                    stack.append(tag)
                elif tag == close_tag:
                    if not stack:
                        # 没有对应的开放标签，说明标签不配对
                        return False
                    stack.pop()
            # 最终栈必须为空，才是严格交替
            return len(stack) == 0


        format_score = 0.0 if if_val else 0.1
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            
            # process the data_item to the token and decode them
            processed_data = self._process_data(data_item=data_item, tokenizer=tokenizer)
            ground_truth, db_id, response_str = processed_data['ground_truth'], processed_data['db_id'], processed_data['response_str']
            
            # reserved for compatibility
            prompt_str, extra_info = processed_data['prompt_str'], processed_data['extra_info']

            score = compute_score_em(response_str, ground_truth, db_id, format_score=format_score)
            scores.append([score])

        return scores
