# NL2SQL Agentic RL 学习备忘录

## 项目整体架构

三阶段流水线：

1. **Agent 轨迹采集**：用教师模型（Qwen2.5-32B）在 BIRD train.json 上生成多轮交互轨迹，构建 SFT 数据集
2. **Agentic SFT 训练**：用轨迹数据微调学生模型（Qwen2.5-Coder-7B），让模型学会 think→act→observe 的交互格式
3. **Agentic RL 训练（GRPO）**：以 SFT 模型为起点，用 BIRD 问题 + 真实数据库执行反馈做强化学习

---

## 阶段一：轨迹采集

- **输入**：`bird/train.json`（9428 条 NL 问题 + DB schema + gold SQL）
- **教师模型**：Qwen2.5-32B-Coder-Instruct，通过 `port 10086` 调用
- **gold SQL 的用途**：BIRD 自带的人工标注答案，用于过滤低质量轨迹（教师模型最终 SQL 答错的轨迹丢掉）
- **产出**：`trajectories/trajectory.jsonl`，每条包含完整的"思考-执行-观察"多轮过程

**为什么不直接用 gold SQL 做 SFT？**

gold SQL 只是最终答案，没有解题过程。SFT 需要的是"轨迹"——遇到报错怎么推理、怎么修改、怎么验证结果。用 gold SQL 训练出来的是一次性生成 SQL 的能力，没有 agentic 交互能力（错了不会改）。

---

## 阶段三：Agentic RL 训练（GRPO）

### RL 数据集

格式（`RL-Factory/preprocess_data.py`）：
```json
{
  "prompt": "<NL问题 + DB schema>",
  "reward_model": {
    "db_id": "movie_platform",
    "ground_truth": "SELECT director_name FROM movies WHERE ..."
  }
}
```

**只有问题和标准答案，没有轨迹。** 轨迹在训练时由当前策略模型在线生成。

来源同样是 BIRD train.json，与阶段一用的是同一批问题，但使用方式完全不同：
- 阶段一：问题 → 教师模型生成示范轨迹 → 学生模型模仿
- 阶段三：问题 → 学生模型自己探索 → 用执行结果奖励更新自身

### GRPO 工作流程

对同一问题，模型并行生成 G 条（如 8 条）轨迹，通过组内相对优势更新模型：

1. 取一批 BIRD 问题
2. 当前策略模型以不同随机性生成 G 条 SQL 轨迹
3. 每条轨迹的最终 SQL 通过 `sql_server`（port 11111）在真实数据库上执行，与 gold SQL 比较
4. GRPO 计算组内相对优势（哪些轨迹比平均好），增大好轨迹的概率，降低差轨迹的概率

### 两个服务端口

| 服务 | 端口 | RL 阶段是否需要 |
|------|------|----------------|
| `bird/sql_server.py` | 11111 | **必须开**，是奖励信号的唯一来源 |
| Qwen2.5-32B-Coder-Instruct | 10086 | **不需要**，当前实现是规则化奖励，不用 LLM 打分 |

奖励计算见 `RL-Factory/envs/nl2sql.py`，调用链：`compute_score_em` → `em_check` → `sync_compare_sql` → `http://127.0.0.1:11111/compare_sql`

---

## 我的工作：GRPO 可以做什么

### 1. 奖励函数设计（最核心）

当前实现（`nl2sql.py`）是最简单的二值奖励 + 格式分。可改进方向：

- **结果奖励细化**：SQL 语法合法但结果错 → 给 0.2；执行结果为空但 gold 不为空 → 额外惩罚 -0.2
- **过程奖励**：每次工具调用成功 → +0.1；重复犯相同错误 → -0.1（`base.py` 已有 `use_process_reward` 开关，待实现）
- **自我修正质量**：第 1 次写错第 2 次改对比写了 4 次才对奖励更高

代码位置：`RL-Factory/envs/nl2sql.py: compute_score_em`

### 2. 训练样本难度过滤

用 SFT 初始模型对每道题采样 G 次，计算通过率：
- pass@G = 1.0（总对）→ 太简单，丢掉
- pass@G = 0.0（总错）→ 太难，丢掉
- 0.1 ≤ pass@G ≤ 0.9 → 保留（有效学习区间）

代码位置：`RL-Factory/preprocess_data.py` 新增过滤逻辑

### 3. 信用分配（Credit Assignment）

**问题**：4 步轨迹最终答对，奖励 +1 如何分配给各步骤？平摊会稀释关键决策 token 的学习信号。

可探索方向：
- **步骤级**：奖励按步骤反向传播，越靠近正确结果的步骤分配越多（如 0.1/0.2/0.3/0.4）
- **片段级**：只对工具调用相关 token 分配信用，过渡性文字不参与
- **关键 token 增强**：`<sql>` 后第一个 token、`<final_sql>` 内容重点强化

代码位置：`RL-Factory/envs/base.py: compute_score` + verl 框架配置

### 4. GRPO 超参调优

- **G（每题轨迹数）**：G 越大信号越准，但显存消耗成比例增加，根据服务器显存找平衡
- **KL 散度惩罚系数**：太小会忘掉 SFT 格式规范，太大 RL 学不到新行为
- **rollout 温度**：影响轨迹多样性

### 优先级

| 优先级 | 工作 |
|--------|------|
| 最高 | 奖励函数：结果奖励细化 + 空结果惩罚 |
| 高 | 训练样本难度过滤 |
| 中 | 信用分配研究 |
| 中 | 过程奖励实现 |
| 低 | GRPO 超参扫描 |

---

## 关键文件速查

| 文件 | 作用 |
|------|------|
| `bird/train.json` | BIRD 训练集（9428条，含 gold SQL） |
| `collect_trajectory.py` | 阶段一轨迹采集脚本 |
| `trajectories/trajectory.jsonl` | 采集到的 SFT 训练轨迹 |
| `RL-Factory/preprocess_data.py` | RL 数据集预处理（生成 parquet） |
| `RL-Factory/envs/nl2sql.py` | NL2SQL 奖励函数实现 |
| `RL-Factory/envs/base.py` | RL 环境基类（信用分配、过程奖励框架） |
| `utils.py` | `sync_compare_sql` → 调用 sql_server port 11111 |
| `bird/sql_server.py` | SQL 执行验证服务，port 11111 |
