experiment_name='acic2016_acquisition_loop'
acquisition_list=("random" "r_PO" "r_TE" "r_sTE")
seed_list=(1 2 3 4 5 6 7 8 9 10)
device="cpu"
subsample='random'
rho_te_list=(0.3 0.7)
pi_setup="confounding"

for acquisition in "${acquisition_list[@]}"; do
    for rho_te in "${rho_te_list[@]}"; do
        for seed in "${seed_list[@]}"; do
            python3  acquisition_loop.py dataset=acic2016 tune_cate_model=True random_seed=$seed experiment_name=$experiment_name acquisition.subsample=$subsample dataset.dataset.rho_TE=$rho_te dataset.dataset.setup_pi=$pi_setup acquisition_metric=$acquisition wandb_log=True device=$device
        done
    done
done

