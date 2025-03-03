acquisition_list=("r_TE")
experiment_name="stop_acquisition_ihdp_appendix_vars"
var_threshold=(0.99 0.95 0.90 0.85 0.80 0.75 0.70)
seed_list=(5)
threshold_variable='sTE'
subsample='random'
device="cuda:2"
alpha=1.0


for seed in "${seed_list[@]}"; do
    for threshold in "${var_threshold[@]}"; do
        for acquisition in "${acquisition_list[@]}"; do
            python3  acquisition_stop.py dataset=ihdp acquisition.threshold_variable=$threshold_variable tune_cate_model=True acquisition.percentile_var_threshold=$threshold acquisition.alpha=$alpha random_seed=$seed experiment_name=$experiment_name acquisition.subsample=$subsample acquisition_metric=$acquisition wandb_log=True device=$device
        done
    done
done