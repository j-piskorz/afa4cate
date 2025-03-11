experiment_name="acic2016_acquisition_stop"
acquisition_list=("random" "r_PO" "r_TE" "r_sTE")
var_threshold=(0.99 0.95 0.90 0.85 0.80 0.75 0.70)
seed_list=(1 2 3 4 5 6 7 8 9 10)
threshold_variable='TE'
subsample='random'
device="cpu"
alpha=1.0
rho_te=0.3
pi_setup="overlap_violation"


for threshold in "${var_threshold[@]}"; do
    for acquisition in "${acquisition_list[@]}"; do
        for seed in "${seed_list[@]}"; do
            python3  acquisition_stop.py dataset=acic2016 acquisition.threshold_variable=$threshold_variable tune_cate_model=True acquisition.percentile_var_threshold=$threshold acquisition.alpha=$alpha random_seed=$seed experiment_name=$experiment_name acquisition.subsample=$subsample acquisition_metric=$acquisition dataset.dataset.rho_TE=$rho_te dataset.dataset.setup_pi=$pi_setup wandb_log=True device=$device
        done
    done
done