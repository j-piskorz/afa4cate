rho_te_list=(0.3 0.7)
lambda_list=(1.0)
pi_setup_list=("confounding")
acquisition_list=("r_TE" "r_PO" "r_sTE" "random")
experiment_name="ultimate_acic2016"
seed_list=(9 10)
subsample='random'
device="cuda:3"

for seed in "${seed_list[@]}"; do
    for rho_te in "${rho_te_list[@]}"; do
        for lambd in "${lambda_list[@]}"; do
            for pi_setup in "${pi_setup_list[@]}"; do
                for acquisition in "${acquisition_list[@]}"; do
                    python3  training_models.py dataset=synthetic tune_cate_model=True random_seed=$seed experiment_name=$experiment_name acquisition.subsample=$subsample dataset.dataset.rho_TE=$rho_te dataset.dataset.lambd=$lambd acquisition_metric=$acquisition wandb_log=True dataset.dataset.setup_pi=$pi_setup device=$device
                done
            done
        done
    done
done