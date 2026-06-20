#!/bin/bash
# GRPO training starting from agent-7b-6656 SFT model
# GPUs: 1, 2 (A40, 46GB each) -- GPU 4 is hardware-faulted (nvidia-smi shows ERR!),
# all other cards are occupied by other users, so only 1,2 are usable.
# sql_server must be running on port 11111

set -e -x

export CUDA_VISIBLE_DEVICES=1,2

# NCCL 配置：避免与其他训练冲突
export MASTER_PORT=29501
# A40 无 NVLink、走 PCIe，开启 P2P 时 NCCL 首次 collective(FSDP sync broadcast)
# 会死锁(两卡 100% 空转、显存上不去)。必须禁用 P2P，走共享内存/主机中转。
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN

# === 防卡死关键修复 ===
# 根分区(/)已用 ~96%，Ray 默认磁盘阈值 0.95 会判定"满"并拒绝创建需要 spill 的对象，
# 导致 FSDP->vLLM 权重传递时 put() 静默挂起、vLLM 永远起不来、显存一直很低。
# 抬高阈值到 0.99（当前 96% 用量 < 99% 即不阻塞）。
export RAY_local_fs_capacity_threshold=0.99
# 切勿设 expandable_segments:True —— vLLM 的 cumem(sleep 模式)有硬断言禁止它，会导致 vLLM 初始化直接失败。
# Ray 临时目录显式放根分区(/home 已 100% 满，不能用)
export RAY_TMPDIR=/tmp/ray

export MODEL_PATH=/var/tmp/lijunhui/models/agent-7b-6656
export REWARD_MODEL_PATH=/home/koujianshang/models/dummy

# /home is FULL — save checkpoints to /var/tmp (root partition, ~65GB free)
export RESULT_DIR=/var/tmp/lijunhui/rl_output

export TRAIN_DATA=/home/koujianshang/agentic-nl2sql/RL-Factory/data/train.parquet
export VAL_DATA=/home/koujianshang/agentic-nl2sql/RL-Factory/data/test.parquet

MAX_CKPTS=1  # disk is tight: keep only the latest 1 checkpoint (~14GB)

mkdir -p $RESULT_DIR

# Background checkpoint cleaner: every 60s, delete old checkpoints keeping only latest $MAX_CKPTS
cleanup_old_ckpts() {
    while true; do
        sleep 60
        CKPT_DIR="$RESULT_DIR/sft_agent7b_improved_reward"
        if [ -d "$CKPT_DIR" ]; then
            # Find step directories (global_step_*), sort by number, remove oldest
            CKPT_DIRS=$(find "$CKPT_DIR" -maxdepth 1 -type d -name "global_step_*" | sort -t_ -k3 -n)
            N_CKPTS=$(echo "$CKPT_DIRS" | grep -c "global_step_" || true)
            if [ "$N_CKPTS" -gt "$MAX_CKPTS" ]; then
                N_DEL=$((N_CKPTS - MAX_CKPTS))
                echo "$CKPT_DIRS" | head -n "$N_DEL" | while read -r dir; do
                    echo "[Cleanup] Removing old checkpoint: $dir"
                    rm -rf "$dir"
                done
            fi
        fi
        # Also report disk usage
        USED=$(du -sh "$RESULT_DIR" 2>/dev/null | cut -f1)
        FREE=$(df -h /var/tmp/ | tail -1 | awk '{print $4}')
        echo "[Disk] RESULT_DIR: $USED, /var/tmp free: $FREE"
    done
}
cleanup_old_ckpts &
CLEANUP_PID=$!
trap "kill $CLEANUP_PID 2>/dev/null" EXIT

cd /home/lijunhui/agentic-nl2sql/RL-Factory

/home/koujianshang/miniconda3/envs/rlfac/bin/python3 -m verl.trainer.main_ppo \
    --config-name=rl_factory_ppo_trainer \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=16 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.state_masking=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.max_turns=3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.env.name=nl2sql \
    actor_rollout_ref.env.mcp_mode=stdio \
    actor_rollout_ref.env.tool_manager=qwen3 \
    actor_rollout_ref.env.enable_thinking=False \
    actor_rollout_ref.env.config_path=envs/configs/mcp_tools.pydata \
    actor_rollout_ref.env.use_process_reward=False \
    reward_rollout.if_use_reward_rollout=False \
    reward_rollout.rollout.tensor_model_parallel_size=1 \
    reward_rollout.rollout.gpu_memory_utilization=0.65 \
    reward_rollout.rollout.model_name=$REWARD_MODEL_PATH \
    reward_rollout.rollout.free_cache_engine=True \
    reward_rollout.rollout.response_length=2048 \
    reward_model.reward_manager=parallel \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='GRPO_nl2sql' \
    trainer.experiment_name='sft_agent7b_improved_reward' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.default_local_dir=$RESULT_DIR \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=200 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 "$@" 2>&1 | tee grpo_sft.log
