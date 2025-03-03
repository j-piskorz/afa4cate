acquisition_list=("r_sTE_entropy")
experiment_name="ultimate_ihdp_new"
seed_list=(1 2 3 4 5 6 7 8 9 10)
device="cuda:3"

for seed in "${seed_list[@]}"; do
    for acquisition in "${acquisition_list[@]}"; do
        python3  training_models.py cate_model=deep_kernel_gp dataset=ihdp_cov tune_cate_model=True random_seed=$seed experiment_name=$experiment_name acquisition.subsample='random' acquisition_metric=$acquisition wandb_log=True device=$device
    done
done