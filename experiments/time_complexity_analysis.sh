experiment_name="acic2016_time_complexity_analysis"
rho_te=0.3
pi_setup='confounding'
acquisition='r_TE'
seed_list=(1 2 3 4 5 6 7 8 9 10)
subsample='random'
n_test_samples=10
device="cuda:0"
num_ones_list=(0 2 4 6 8 10 12 14 16 18)

for seed in "${seed_list[@]}"; do
    for num_ones in "${num_ones_list[@]}"; do
        python3  acquisition_loop.py dataset=acic2016 dataset.num_ones=$num_ones tune_cate_model=True random_seed=$seed experiment_name=$experiment_name acquisition.subsample=$subsample acquisition.n_test_samples=$n_test_samples dataset.dataset.rho_TE=$rho_te acquisition_metric=$acquisition wandb_log=True dataset.dataset.setup_pi=$pi_setup device=$device
    done
done