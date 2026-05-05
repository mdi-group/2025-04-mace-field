
# Recommended after regenerating MP-Ferroelectrics-{train,valid,test}.xyz with:
#   python get_ferroelectric_dataset.py --out MP-Ferroelectrics.xyz --no-allow-branch-cross-split

torchrun --standalone --nproc_per_node="gpu" -m mace.cli.run_train \
    --name="MACE-Field-MP-Ferroelectrics" \
    --train_file="MP-Ferroelectrics-train.xyz" \
    --test_file="MP-Ferroelectrics-test.xyz" \
    --valid_file="MP-Ferroelectrics-valid.xyz" \
    --E0s="average" \
    --loss='universal_field' \
    --energy_weight=100.0 \
    --forces_weight=10.0 \
    --stress_weight=0.1 \
    --polarization_weight=100.0 \
    --polarization_loss_mode=normalized_metric \
    --polarization_loss_scale=1.0 \
    --becs_weight=0.0 \
    --polarizability_weight=0.0 \
    --compute_polarization=True \
    --compute_becs=False \
    --compute_polarizability=False \
    --compute_forces=True \
    --compute_stress=True \
    --model="MACEField" \
    --num_channels=128 \
    --num_workers=8 \
    --lr=0.001 \
    --ema \
    --ema_decay=0.995 \
    --scheduler_patience=5 \
    --max_num_epochs=300 \
    --patience=100 \
    --amsgrad \
    --distributed \
    --device="cuda" \
    --enable_cueq True \
    --seed=23 \
    --default_dtype="float64" \
    --save_cpu \
    --batch_size 2 \
    --valid_batch_size 2 \
    --plot=True \
    --restart_latest \
