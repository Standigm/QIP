
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 -T dgx env MASTER_PORT=45100 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MT0_gnn.yaml seed=8252 name=gatv2_MT0_1_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 -T dgx env MASTER_PORT=45110 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MT0_gnn.yaml seed=8262 name=gatv2_MT0_2_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 -T dgx env MASTER_PORT=45120 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MT0_gnn.yaml seed=8272 name=gatv2_MT0_3_new trainer.max_steps=20000 trainer.devices=2

# MT- Gatv2 HAD
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 -T dgx env MASTER_PORT=45000 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MT0_gnn.yaml seed=8252 name=gatv2_MTHAD_1 trainer.max_steps=10000 trainer.devices=2 system.encoder_config.state_path=/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gatv2_PTHAD/2023-11-15/12-11-51/checkpoints/checkpoint_000115128/encoder.pt
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 -T dgx env MASTER_PORT=45140 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MT0_gnn.yaml seed=8262 name=gatv2_MTHAD_2 trainer.max_steps=10000 trainer.devices=2 system.encoder_config.state_path=/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gatv2_PTHAD/2023-11-15/12-11-51/checkpoints/checkpoint_000115128/encoder.pt
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 -T dgx env MASTER_PORT=45150 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MT0_gnn.yaml seed=8272 name=gatv2_MTHAD_3 trainer.max_steps=10000 trainer.devices=2 system.encoder_config.state_path=/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gatv2_PTHAD/2023-11-15/12-11-51/checkpoints/checkpoint_000115128/encoder.pt


# MT - GINE HAD

HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 -T dgx env MASTER_PORT=45000 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MT0_gine.yaml seed=8252 name=gine_MTHAD_1 trainer.max_steps=10000 trainer.devices=2 system.encoder_config.state_path=/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gine_PTHAD/2023-11-15/12-17-46/checkpoints/checkpoint_000108984/encoder.pt
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 -T dgx env MASTER_PORT=45140 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MT0_gine.yaml seed=8262 name=gine_MTHAD_2 trainer.max_steps=10000 trainer.devices=2 system.encoder_config.state_path=/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gine_PTHAD/2023-11-15/12-17-46/checkpoints/checkpoint_000108984/encoder.pt
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 -T dgx env MASTER_PORT=45150 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MT0_gine.yaml seed=8272 name=gine_MTHAD_3 trainer.max_steps=10000 trainer.devices=2 system.encoder_config.state_path=/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gine_PTHAD/2023-11-15/12-17-46/checkpoints/checkpoint_000108984/encoder.pt



# MT- Gatv2 HAD
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 -T dgx env MASTER_PORT=45000 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MT0_gnn.yaml seed=8252 name=gatv2_MT0_1 trainer.max_steps=10000 trainer.devices=2 
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 -T dgx env MASTER_PORT=45140 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MT0_gnn.yaml seed=8262 name=gatv2_MT0_2 trainer.max_steps=10000 trainer.devices=2 
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 -T dgx env MASTER_PORT=45150 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MT0_gnn.yaml seed=8272 name=gatv2_MT0_3 trainer.max_steps=10000 trainer.devices=2 


# MT - GINE HAD

HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 -T dgx env MASTER_PORT=45200 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MT0_gine.yaml seed=8252 name=gine_MT0_1 trainer.max_steps=10000 trainer.devices=2 
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 -T dgx env MASTER_PORT=45210 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MT0_gine.yaml seed=8262 name=gine_MT0_2 trainer.max_steps=10000 trainer.devices=2 
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 -T dgx env MASTER_PORT=45220 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MT0_gine.yaml seed=8272 name=gine_MT0_3 trainer.max_steps=10000 trainer.devices=2 
