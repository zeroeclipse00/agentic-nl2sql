import re
import json
import asyncio
import requests
from typing import List
from omegaconf import OmegaConf
from envs.tool_manager.base_manager import ToolManager


class NL2SQLManager(ToolManager):
    """Tool manager for NL2SQL agentic format.

    Handles the custom <sql>/<sql_exec_result>/<final_sql> interaction format
    used by SFT models trained on NL2SQL trajectories, instead of standard
    <tool_call> format used by qwen3/qwen2_5 managers.
    """

    def __init__(self, verl_config):
        if isinstance(verl_config, dict):
            verl_config = OmegaConf.create(verl_config)
        self._batch_db_ids: List[str] = []
        super().__init__(verl_config)

    def set_batch_db_ids(self, db_ids: List[str]):
        self._batch_db_ids = list(db_ids)

    @property
    def all_tools(self):
        return self.tool_map

    def _build_tools(self):
        self.functions = []

    # ── prompt formatting ────────────────────────────────────────────

    def get_prompt(self, input_data, tokenizer, mode='initial', add_generation_prompt=True):
        assert mode in ['initial', 'tool_call', 'assistant_response']

        if mode == 'initial':
            if isinstance(input_data, str):
                input_data = [{'role': 'user', 'content': input_data}]
            elif isinstance(input_data, list) and input_data and isinstance(input_data[0], str):
                input_data = [{'role': 'user', 'content': ''.join(input_data)}]
            return tokenizer.apply_chat_template(
                conversation=input_data,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        if mode == 'tool_call':
            if isinstance(input_data, list):
                parts = []
                for item in input_data:
                    if isinstance(item, dict):
                        parts.append(item.get('content', ''))
                    else:
                        parts.append(str(item))
                return ''.join(parts)
            return input_data if isinstance(input_data, str) else str(input_data)

        return input_data if isinstance(input_data, str) else str(input_data)

    # ── response parsing ─────────────────────────────────────────────

    def parse_response(self, response_content: str):
        if self._has_final_sql(response_content):
            return 'answer', response_content

        sql = self._extract_last_sql(response_content)
        if sql is not None:
            return 'actions', [{'name': 'sql', 'args': json.dumps({'sql': sql}, ensure_ascii=False)}]

        return 'answer', response_content

    def parse_end_flag(self, response_content: str):
        if self._has_final_sql(response_content):
            return True, response_content
        return False, None

    @staticmethod
    def _has_final_sql(text: str) -> bool:
        return '<final_sql>' in text

    @staticmethod
    def _extract_last_sql(text: str):
        matches = list(re.finditer(r'<sql>(.*?)</sql>', text, re.DOTALL))
        if matches:
            return matches[-1].group(1).strip()
        return None

    # ── SQL execution ────────────────────────────────────────────────

    def _exec_sql(self, sql: str, db_id: str) -> str:
        payload = {'db_id': db_id, 'sql': sql, 'timeout': 60}
        try:
            resp = requests.post(
                'http://127.0.0.1:11111/execute_sql',
                json=payload, timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                result = str(data.get('result', ''))
                if len(result) > 500:
                    result = result[:500] + '... (Omitted for brevity)'
                if data.get('status') == 1:
                    return f'<sql_exec_result>✅ SQL执行成功，结果：{result}</sql_exec_result>'
                return f'<sql_exec_result>❌ SQL执行失败，原因：{result}</sql_exec_result>'
            return '<sql_exec_result>❌ SQL执行失败，原因：请求失败</sql_exec_result>'
        except Exception as e:
            return f'<sql_exec_result>❌ SQL执行失败，原因：{e}</sql_exec_result>'

    # ── action execution (sync, used by env.step via FSDP path) ──────

    def execute_actions(self, responses: List[str]):
        actions, tool_results = [], []
        for i, response in enumerate(responses):
            action, parsed = self.parse_response(response)
            actions.append(action)

            if action == 'answer':
                tool_results.append(parsed)
            elif action == 'actions':
                sql = json.loads(parsed[0]['args'])['sql']
                db_id = self._batch_db_ids[i] if i < len(self._batch_db_ids) else ''
                result = self._exec_sql(sql, db_id)
                tool_results.append([{'role': 'tool', 'content': result}])

        return actions, tool_results

    # ── async execution (used by chat_scheduler callback) ────────────

    async def execute_all_tools(self, actions, tool_list):
        results = []
        for action, tools in zip(actions, tool_list):
            if action == 'answer':
                results.append({'role': 'assistant', 'content': tools if isinstance(tools, str) else str(tools)})
            elif action == 'actions':
                sql = json.loads(tools[0]['args'])['sql']
                db_id = self._batch_db_ids[0] if self._batch_db_ids else ''
                result = await asyncio.to_thread(self._exec_sql, sql, db_id)
                results.append([{'role': 'tool', 'content': result}])
            else:
                results.append({'role': 'assistant', 'content': str(tools)})
        return results
