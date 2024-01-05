HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32 -T dgx -W dgx801 env MASTER_PORT=45600 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTR.yaml seed=8282
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32 -T dgx -W dgx801 env MASTER_PORT=45610 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTR_only.yaml seed=8282 
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32 -T dgx -W dgx802 env MASTER_PORT=45620 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTHADR.yaml seed=8282
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32 -T dgx -W dgx802 env MASTER_PORT=45630 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTHADR_only.yaml seed=8282


HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32 -T dgx env MASTER_PORT=47000 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTHADRS.yaml seed=8282
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32 -T dgx env MASTER_PORT=47010 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTS.yaml seed=8282
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 4 -C 32 -T dgx env MASTER_PORT=47020 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/pretraining/gps_PTRS.yaml seed=8282
