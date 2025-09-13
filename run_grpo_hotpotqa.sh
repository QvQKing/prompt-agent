export BASE_MODEL='/home/luohaoran/wenjin/new/Agent-R1/Qwen/Qwen3-4B'
# export BASE_MODEL='/home/luohaoran/wenjin/new/Agent-R1/Qwen/Qwen2.5-1.5B-Instruct'
# export BASE_MODEL='Qwen/Qwen2.5-1.5B-Instruct'
export PROJECT_NAME='Prompt-R1'
export EXPERIMENT_NAME=grpo-qwen3-4b-gpt4

# 只用 GPU 2 和 3
export CUDA_VISIBLE_DEVICES=0,1,2,3

set -x

ray stop
ray start --head

python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    data.train_files=['data/hotpotqa/train.parquet'] \
    data.val_files=['data/hotpotqa/validation.parquet'] \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.max_response_length_single_turn=1024 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.stop_token_ids=[151645] \
    actor_rollout_ref.rollout.stop=[] \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n_repeat=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=160 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    trainer.log_val_generations=0 \
    tool.max_turns=5 \
    tool.tools=['prompt'] \
    tool.max_tool_response_length=2048 $@