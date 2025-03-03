rho_te_list=(0.3)
lambda_list=(1.0)
pi_setup_list=("confounding")
acquisition_list=("r_TE")
experiment_name="acic2016_time_complexity"
seed_list=(4 5 6 7 8 9 10)
subsample='random'
n_test_samples=10
device="cuda:3"
num_ones_list=(0 2 4 6 8 10 12 14 16 18)

for seed in "${seed_list[@]}"; do
    for rho_te in "${rho_te_list[@]}"; do
        for lambd in "${lambda_list[@]}"; do
            for pi_setup in "${pi_setup_list[@]}"; do
                for acquisition in "${acquisition_list[@]}"; do
                    for num_ones in "${num_ones_list[@]}"; do
                        python3  training_models.py dataset=synthetic dataset.num_ones=$num_ones tune_cate_model=True random_seed=$seed experiment_name=$experiment_name acquisition.subsample=$subsample acquisition.n_test_samples=$n_test_samples dataset.dataset.rho_TE=$rho_te dataset.dataset.lambd=$lambd acquisition_metric=$acquisition wandb_log=True dataset.dataset.setup_pi=$pi_setup device=$device
                    done
                done
            done
        done
    done
done