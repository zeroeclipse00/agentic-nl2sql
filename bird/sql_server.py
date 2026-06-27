# gunicorn     -w 4     --threads 32     --max-requests 1000     --max-requests-jitter 10     --timeout 300     -b 0.0.0.0:11111     sql_server:app

import sqlite3
import os
import multiprocessing as mp
from collections import Counter
import re
import time
import sys
from flask import Flask, request, jsonify
from functools import wraps
import json
import traceback
import argparse

# 【核心修复 1】强制使用 spawn 模式，防止子进程继承 Flask 主进程的庞大内存
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

db_prefix = os.environ['HOME'] + "/autodl-tmp/bird_db"

app = Flask(__name__)

def execute_sql_in_subprocess(sql: str, db_path: str, db_id: str, query_id: int):
    """
    Execute a SQL query with caching in a subprocess.
    """
    try:
        abs_path = os.path.abspath(db_path)
        db_uri = f"file:{abs_path}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True)
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.close()
        return result
    except Exception as e:
        # 保持原有逻辑不变
        try:
            conn.close()
        except:
            pass
        return f"Execution error: {e}"

# --- 新增的 Worker 函数：用于隔离内存 ---
def _worker_execute_single(sql, db_path, db_id, query_id, queue):
    res = execute_sql_in_subprocess(sql, db_path, db_id, query_id)
    queue.put(res)

def _worker_compare(predicted_sql, ground_truth, db_path, db_id, query_id, queue):
    # 在子进程中获取数据
    predicted_res = execute_sql_in_subprocess(predicted_sql, db_path, db_id, query_id)
    ground_truth_res = execute_sql_in_subprocess(ground_truth, db_path, db_id, query_id)
    
    # 【核心修复 2】评测逻辑原封不动，但移入子进程执行，防止海量结果传回主进程导致 OOM
    if isinstance(predicted_res, str) or isinstance(ground_truth_res, str):
        error_msg = predicted_res if isinstance(predicted_res, str) else ground_truth_res
        queue.put((0, f"语法错误: {error_msg}"))
        return

    # 转换为元组集合（列表不可哈希，转tuple才能用于set判断）
    pred_set = set(map(tuple, predicted_res))
    gt_set = set(map(tuple, ground_truth_res))

    # 完全保留原有比对逻辑 + 新增子集判断
    if pred_set == gt_set:
        # 结果完全一致
        queue.put((1, "正确"))
    elif gt_set.issubset(pred_set):
        # 核心修改：真实结果是预测结果的子集 → 预测存在多余列
        queue.put((0, "存在多余列"))
    else:
        # 其他不匹配情况
        queue.put((0, "结果不匹配"))

def _worker_compare_old(predicted_sql, ground_truth, db_path, db_id, query_id, queue):
    # 在子进程中获取数据
    predicted_res = execute_sql_in_subprocess(predicted_sql, db_path, db_id, query_id)
    ground_truth_res = execute_sql_in_subprocess(ground_truth, db_path, db_id, query_id)
    
    # 【核心修复 2】评测逻辑原封不动，但移入子进程执行，防止海量结果传回主进程导致 OOM
    if isinstance(predicted_res, str) or isinstance(ground_truth_res, str):
        error_msg = predicted_res if isinstance(predicted_res, str) else ground_truth_res
        queue.put((0, f"语法错误: {error_msg}"))
        return

    # 完全保留原有的比对逻辑
    if set(map(tuple, predicted_res)) == set(map(tuple, ground_truth_res)):
        queue.put((1, "正确"))
    else:
        queue.put((0, "结果不匹配"))

def execute_single_sql(predicted_sql: str, db_id: str, query_id: int = 0, timeout: int = 60):
    db_path = os.path.join(db_prefix, f"./{db_id}/{db_id}.sqlite")
    
    # 【核心修复 3】废弃 Pool，使用单次 Process + Queue，配合安全的超时读取防止死锁
    queue = mp.Queue()
    p = mp.Process(target=_worker_execute_single, args=(predicted_sql, db_path, db_id, query_id, queue))
    p.start()
    
    res_data = None
    got_result = False
    try:
        # 优先通过 queue 等待结果，防止大数据造成管道阻塞和死锁
        res_data = queue.get(timeout=timeout)
        got_result = True
    except Exception:
        pass

    if p.is_alive():
        p.terminate()
        p.join()
        if not got_result:
            return 0, f"Execution error: Timeout after {timeout} seconds"
    else:
        p.join()
        
    if got_result:
        if isinstance(res_data, str):
            return 0, res_data
        else:
            return 1, res_data
    else:
        return 0, "Execution error: Process crashed silently"

def compare_sql(predicted_sql: str, ground_truth: str, db_id: str, query_id: int=0, timeout: int = 60):
    """
    Execute predicted and ground truth SQL queries with caching, 
    comparing results for correctness.
    """
    db_path = os.path.join(db_prefix, f"{db_id}/{db_id}.sqlite")

    queue = mp.Queue()
    p = mp.Process(target=_worker_compare, args=(predicted_sql, ground_truth, db_path, db_id, query_id, queue))
    p.start()
    
    res_data = None
    got_result = False
    try:
        res_data = queue.get(timeout=timeout)
        got_result = True
    except Exception:
        pass

    if p.is_alive():
        p.terminate()
        p.join()
        if not got_result:
            return 0, f"Execution error: Timeout after {timeout} seconds"
    else:
        p.join()
        
    if got_result:
        return res_data
    else:
        return 0, "Execution error: Process crashed silently"


# 添加错误处理装饰器
def handle_exceptions(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            error_traceback = traceback.format_exc()
            return jsonify({
                'status': 'error',
                'message': str(e),
                'traceback': error_traceback
            }), 500
    return decorated_function

# Flask 路由 - 执行单个 SQL
@app.route('/execute_sql', methods=['POST'])
@handle_exceptions
def api_execute_single_sql():
    data = request.json
    if not data or not all(k in data for k in ['sql', 'db_id']):
        return jsonify({'status': 'error', 'message': '缺少必要参数'}), 400
    
    sql = data['sql']
    db_id = data['db_id']
    query_id = data.get('query_id', 0)
    timeout = data.get('timeout', 60)
    
    try:
        status, result = execute_single_sql(sql, db_id, query_id, timeout)
        return jsonify({
            'status': status,
            'result': result
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Execution error: {str(e)}'}), 500

# Flask 路由 - 比较 SQL
@app.route('/compare_sql', methods=['POST'])
@handle_exceptions
def api_compare_sql():
    data = request.json
    if not data or not all(k in data for k in ['predicted_sql', 'ground_truth', 'db_id']):
        return jsonify({'status': 'error', 'message': 'Missing required parameters'}), 400
    
    predicted_sql = data['predicted_sql']
    ground_truth = data['ground_truth']
    db_id = data['db_id']
    query_id = data.get('query_id', 0)
    timeout = data.get('timeout', 60)
    
    try:
        status, result = compare_sql(predicted_sql, ground_truth, db_id, query_id, timeout)
        return jsonify({
            'status': status,
            'result': result
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Execution error: {str(e)}'}), 500

if __name__ == '__main__':
    # 设置线程模式以支持并发请求
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=6789)
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port, threaded=True)
