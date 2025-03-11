experiment_name='ihdp_acquisition_loop'
acquisition_list=("random" "r_PO" "r_TE" "r_sTE")
seed_list=(1 2 3 4 5 6 7 8 9 10)
device="cpu"
subsample='random'


for acquisition in "${acquisition_list[@]}"; do
    for seed in "${seed_list[@]}"; do
        python3 acquisition_loop.py dataset=ihdp tune_cate_model=True random_seed=$seed experiment_name=$experiment_name acquisition.subsample=$subsample acquisition_metric=$acquisition wandb_log=True device=$device
    done
done

