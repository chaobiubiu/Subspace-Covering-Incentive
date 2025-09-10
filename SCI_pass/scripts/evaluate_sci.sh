# Pass
env="Pass-v0"
ind=0
nagt=2
model_loc="/home/lc/code/SCI_MACE/scripts/logs/results/Pass-v0/0/rmappo/SCI_hierarchical-0402_113435-seed-44--self_coef-10.0--other_coef-10.0--ir_coef-10.0--hierarchical-True--step_interval-20--epoch_ratio-0.7/models/cp_1500"

# SecretRoom
# env="SecretRoom-v0"
# ind=20
# nagt=2

# MultiRoom
# env="SecretRoom-v0"
# ind=33
# nagt=3

CUDA_VISIBLE_DEVICES=0 python ../src/eval.py --env_name ${env} --map_ind ${ind} --n_agents ${nagt} \
--num_env_steps 80000000 --use_eval --n_rollout_threads 1 --n_eval_rollout_threads 1 --eval_interval 100 --save_interval 500 \
--algorithm_name rmappo --use_recurrent_policy --entropy_coef 0.05 \
--novel_type 3 --self_coef 1.0 --other_coef 1.0 \
--use_hdd --ir_coef 1.0 --hdd_count --hdd_count_window 10 --discrete_novel_in_hd \
--save_hdd_count --run_dir logs --experiment_name SCI_new_evaluate --model_dir ${model_loc}