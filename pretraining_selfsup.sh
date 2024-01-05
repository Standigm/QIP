HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32 -T dgx  env MASTER_PORT=45600 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTS.yaml seed=8282
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32 -T dgx  env MASTER_PORT=45610 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTRS.yaml seed=8282
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32 -T dgx -W dgx801 env MASTER_PORT=45620 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTHADS.yaml seed=8282
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32 -T dgx -W dgx802 env MASTER_PORT=45630 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTHADRS.yaml seed=8282 


HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32   env MASTER_PORT=45100 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTH.yaml seed=8282
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32   env MASTER_PORT=45110 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTHA.yaml seed=8282
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32   env MASTER_PORT=45120 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTHD.yaml seed=8282
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32   env MASTER_PORT=45130 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTHAD.yaml seed=8282

HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32  env MASTER_PORT=45140 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTR.yaml seed=8282
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32  env MASTER_PORT=45150 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTR_only.yaml seed=8282
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 8 -C 32 -T dgx env MASTER_PORT=45160 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTRS.yaml seed=8282 trainer.devices=8
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32  env MASTER_PORT=45170 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTS.yaml seed=8282

HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32 -T dgx -W dgx801 env MASTER_PORT=45620 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTHADS.yaml seed=8282
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 8 -C 32 -T dgx -W dgx802 env MASTER_PORT=45630 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTHADRS.yaml seed=8282 trainer.devices=8

HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32 -T dgx -W dgx801  env MASTER_PORT=45600 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTHADR.yaml seed=8282
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32 -T dgx -W dgx802 env MASTER_PORT=45600 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTHADR_only.yaml seed=8282


HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32 -T dgx env MASTER_PORT=45630 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gat_PTHAD.yaml seed=8282
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32 -T dgx env MASTER_PORT=45640 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gine_PTHAD.yaml seed=8282
