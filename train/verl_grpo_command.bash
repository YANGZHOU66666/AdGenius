# VeRL-GRPO训练指令
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$HOME/data/style_imitation/sft_dataset_final.parquet \
 data.val_files=$HOME/data/style_imitation/val_dataset_final.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=256 \
 data.max_response_length=512 \
 data.return_raw_chat=True \
 actor_rollout_ref.model.path=/root/autodl-tmp/Qwen3-4B-Instruct-2507-final \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=16 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.n=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=/root/autodl-tmp/Qwen3-4B-Instruct-2507 \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 algorithm.adv_estimator=grpo \
 trainer.logger=[console,wandb] \
 trainer.project_name='verl_grpo_elemental_equation' \
 trainer.experiment_name='qwen3_4b_grpo' \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=2 \
 trainer.nnodes=1 \
 trainer.save_freq=1 \
 trainer.test_freq=1 \
 reward_model.enable=True \
 reward_model.model.path=/root/autodl-tmp/Skywork-Reward-V2-Qwen3-4B-finetuned \
 reward_model.micro_batch_size_per_gpu=8 \
 trainer.total_epochs=1 2>&1 | tee verl_grpo_qwen3_4B.log

# merge指令
python3 -m verl.model_merger merge \
--backend fsdp \
--local_dir checkpoints/verl_grpo_elemental_equation/qwen3_4b_grpo/global_step_2/actor \
--target_dir /root/autodl-tmp/qwen3_4b_instruct_2507_grpo