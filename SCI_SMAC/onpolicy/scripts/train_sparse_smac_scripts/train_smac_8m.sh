#!/bin/sh
env="SparseStarCraft2"
map="8m"
algo="rmappo"
exp="sci_test"

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}"

CUDA_VISIBLE_DEVICES=5 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
--num_env_steps 10000000 --ppo_epoch 15 --use_value_active_masks --use_eval --eval_episodes 32 --state_novel_coef 0.1 \
--ir_coef 0.0 --num_objects 3
