# if [ ! -f data/field_selected_configs.extxyz ]; then
#     torchrun --standalone --nproc_per_node="gpu" -m mace.cli.fine_tuning_select \
#         --configs_pt mp_traj_combined.xyz \
#         --configs_ft field-data.xyz \
#         --num_samples 10000 \
#         --subselect fps \
#         --model mace-mp-0b3-medium.model \
#         --output mp_traj_selected.xyz \
#         --filtering_type combinations \
#         --head_pt pt_head \
#         --head_ft target_head \
#         --weight_pt 1.0 \
#         --weight_ft 1.0 
# fi

torchrun --standalone --nproc_per_node="gpu" ../mace/cli/run_train.py \
    --config "config.yaml"
    