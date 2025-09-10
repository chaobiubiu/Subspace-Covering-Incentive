#!/bin/sh
env="Grid"
scenario="pass"
num_agents=2
algo="sci" #"mappo" "ippo"
exp="new_state_count_doi_300_test"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python ../train/train_pass.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} --share_policy \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 300 --num_env_steps 50000000 \
    --ppo_epoch 10 --gain 0.01 --entropy_coef 0.05 --bonus_coef 1.0 --lr 7e-4 --critic_lr 7e-4
done