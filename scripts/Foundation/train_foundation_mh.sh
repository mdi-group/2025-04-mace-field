# torchrun --standalone --nproc-per-node="1" -m mace.cli.fine_tuning_select \
#   --configs_pt data/replay-data-mh-1-omat-pbe.xyz \
#   --configs_ft data/MP-Dielectrics-and-Ferroelectrics.xyz \
#   --num_samples 10000 \
#   --subselect fps \
#   --model mace-mh-0.model \
#   --output data/subselected-replay-data-mh-0-omat-pbe.xyz \
#   --filtering_type combinations \
#   --head_pt omat_pbe \
#   --head_ft dielectric \
#   --weight_pt 1.0 \
#   --weight_ft 1.0 \
#   --device cuda

torchrun --standalone --nproc-per-node="gpu" -m mace.cli.run_train \
  --name="MACEField-omat-dielectric-2" \
  --heads "{'mp-dielectric': {'train_file': '../Dielectrics/MP-Dielectrics-filtered-train.xyz', 'valid_file': '../Dielectrics/MP-Dielectrics-filtered-valid.xyz', 'test_file': '../Dielectrics/MP-Dielectrics-filtered-test.xyz'}, 'mp-ferroelectric': {'train_file': '../Ferroelectrics/MP-Ferroelectrics-train.xyz', 'valid_file': '../Ferroelectrics/MP-Ferroelectrics-valid.xyz', 'test_file': '../Ferroelectrics/MP-Ferroelectrics-test.xyz'}}" \
  --foundation_model mace-mh-0.model \
  --foundation_head omat_pbe \
  --pt_train_file data/subselected-replay-data-mh-0-omat-pbe.xyz \
  --multiheads_finetuning True \
  --pseudolabel_replay=True \
  --E0s "foundation" \
  --model="MACEField" \
  --loss="universal_field" \
  --error_table="PerAtomFieldRMSE" \
  --energy_weight=1.0 \
  --forces_weight=10.0 \
  --polarization_weight=10.0 \
  --becs_weight=100.0 \
  --polarizability_weight=10.0 \
  --compute_forces=True \
  --compute_stress=True \
  --compute_polarization=True \
  --polarization_loss_mode=normalized_metric \
  --polarization_loss_scale=1.0 \
  --compute_becs=True \
  --compute_polarizability=True \
  --num_workers 8 \
  --distributed \
  --enable_cueq=True \
  --device="cuda" \
  --plot=True \
  --batch_size 1 \
  --valid_batch_size 1 \
  --max_num_epochs 300 \
  --patience 50 \
  --default_dtype "float32" \
  --restart_latest