acquisition_list=("r_TE")
experiment_name="stop_acquisition_acic2016_appendix"
var_threshold=(0.9)
seed_list=(5)
rho_TE=0.3
subsample='random'
setup_pi="overlap_violation"
device="cuda:1"
alpha=1.0


for seed in "${seed_list[@]}"; do
    for threshold in "${var_threshold[@]}"; do
        for acquisition in "${acquisition_list[@]}"; do
            python3  acquisition_stop.py dataset=synthetic tune_cate_model=True acquisition.percentile_var_threshold=$threshold acquisition.alpha=$alpha random_seed=$seed experiment_name=$experiment_name acquisition.subsample=$subsample acquisition_metric=$acquisition dataset.dataset.rho_TE=$rho_TE dataset.dataset.setup_pi=$setup_pi wandb_log=True device=$device
        done
    done
done