acquisition_list=("r_PO" "r_WPO" "r_TE" "r_TE_total" "random")
lambda_list=(1.0 4.0)
rho_list=(0.3 0.7)
seed_list=(1)
experiment_name="fixed_acic2016"

for seed in "${seed_list[@]}"; do
    for lambd in "${lambda_list[@]}"; do
        for rho_TE in "${rho_list[@]}"; do
            for acquisition in "${acquisition_list[@]}"; do
                python3  training_models.py experiment_name=$experiment_name random_seed=$seed dataset.dataset.setup_mu='A' dataset.dataset.rho_TE=$rho_TE dataset.dataset.lambd=$lambd acquisition_metric=$acquisition wandb_log=True dataset.dataset.setup_pi='confounding' device='cuda:0'
            done
        done
    done
done