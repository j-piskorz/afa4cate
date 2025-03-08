experiment_name="acic2016_acquisition_stop"
acquisition_list=("random")
var_threshold=(0.99)
seed_list=(1)
threshold_variable='TE'
subsample='random'
device="cuda:0"
alpha=1.0
rho_te=0.7
pi_setup="overlap_violation"


for threshold in "${var_threshold[@]}"; do
    for acquisition in "${acquisition_list[@]}"; do
        for seed in "${seed_list[@]}"; do
            python3  acquisition_stop.py dataset=acic2016 acquisition.threshold_variable=$threshold_variable tune_cate_model=True acquisition.percentile_var_threshold=$threshold acquisition.alpha=$alpha random_seed=$seed experiment_name=$experiment_name acquisition.subsample=$subsample acquisition_metric=$acquisition dataset.dataset.rho_TE=$rho_te dataset.dataset.setup_pi=$pi_setup wandb_log=True device=$device
        done
    done
done