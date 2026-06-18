set -x

nproc_per_node=6
save_path=$HOME/models/agent-7b

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/agentic-nl2sql/RL-Factory/data/train.parquet \
    data.val_files=$HOME/agentic-nl2sql/RL-Factory/data/test.parquet \
    data.prompt_key=prompt \
    data.response_key=answer \
    data.max_length=9216 \
    data.truncation=right \
    data.train_batch_size=6 \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=$HOME/models/Qwen2.5-Coder-7B-Instruct \
    optim.lr=5e-6 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=nl2sql-agent-sft \
    trainer.experiment_name=qwen-7b \
    trainer.total_epochs=1 \
    trainer.logger=['console','swanlab'] \
    trainer.default_hdfs_dir=null $@
