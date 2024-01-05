# MT-0
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47800 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MT0.yaml seed=8252 name=momentum_gps_MT0_1_20k trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47810 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MT0.yaml seed=8262 name=momentum_gps_MT0_2_20k trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47820 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MT0.yaml seed=8272 name=momentum_gps_MT0_3_20k trainer.max_steps=20000 trainer.devices=2
# MT- H
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47830 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTH.yaml seed=8252 name=momentum_gps_MTH_1_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47840 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTH.yaml seed=8262 name=momentum_gps_MTH_2_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47850 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTH.yaml seed=8272 name=momentum_gps_MTH_3_new trainer.max_steps=20000 trainer.devices=2
# MT- HA
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47860 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTHA.yaml seed=8252 name=momentum_gps_MTHA_1_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47870 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTHA.yaml seed=8262 name=momentum_gps_MTHA_2_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47880 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTHA.yaml seed=8272 name=momentum_gps_MTHA_3_new trainer.max_steps=20000 trainer.devices=2
# MT- HAD
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47890 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTHD.yaml seed=8252 name=momentum_gps_MTHD_1_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47900 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTHD.yaml seed=8262 name=momentum_gps_MTHD_2_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47910 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTHD.yaml seed=8272 name=momentum_gps_MTHD_3_new trainer.max_steps=20000 trainer.devices=2
# MT- HD
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47920 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTHAD.yaml seed=8252 name=momentum_gps_MTHAD_1_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47930 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTHAD.yaml seed=8262 name=momentum_gps_MTHAD_2_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47940 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTHAD.yaml seed=8272 name=momentum_gps_MTHAD_3_new trainer.max_steps=20000 trainer.devices=2
# MT-R
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47950 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTR.yaml seed=8252 name=momentum_gps_MTR_1_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47960 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTR.yaml seed=8262 name=momentum_gps_MTR_2_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47970 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTR.yaml seed=8272 name=momentum_gps_MTR_3_new trainer.max_steps=20000 trainer.devices=2



#이렇게 반반 나누기
# MT-HADR
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47800 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTHADR.yaml seed=8252 name=momentum_gps_MTHADR_1_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47810 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTHADR.yaml seed=8262 name=momentum_gps_MTHADR_2_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47820 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTHADR.yaml seed=8272 name=momentum_gps_MTHADR_3_new trainer.max_steps=20000 trainer.devices=2
# MT-S
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=48000 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTS.yaml seed=8252 name=momentum_gps_MTS_1_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=48010 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTS.yaml seed=8262 name=momentum_gps_MTS_2_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=48020 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTS.yaml seed=8272 name=momentum_gps_MTS_3_new trainer.max_steps=20000 trainer.devices=2
# MT-RS
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=48030 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTRS.yaml seed=8252 name=momentum_gps_MTRS_1_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=48040 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTRS.yaml seed=8262 name=momentum_gps_MTRS_2_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=48050 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTRS.yaml seed=8272 name=momentum_gps_MTRS_3_new trainer.max_steps=20000 trainer.devices=2
# MT-HADRS
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=48060 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTHADRS.yaml seed=8252 name=momentum_gps_MTHADRS_1_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=48070 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTHADRS.yaml seed=8262 name=momentum_gps_MTHADRS_2_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=48080 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTHADRS.yaml seed=8272 name=momentum_gps_MTHADRS_3_new trainer.max_steps=20000 trainer.devices=2

# MT-R_only
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47950 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTR_only.yaml seed=8252 name=momentum_gps_MTR_only_1_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47960 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTR_only.yaml seed=8262 name=momentum_gps_MTR_only_2_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47970 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTR_only.yaml seed=8272 name=momentum_gps_MTR_only_3_new trainer.max_steps=20000 trainer.devices=2

# MT-HADR_only
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47830 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTHADR_only.yaml seed=8252 name=momentum_gps_MTHADR_only_1_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47840 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTHADR_only.yaml seed=8262 name=momentum_gps_MTHADR_only_2_new trainer.max_steps=20000 trainer.devices=2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=47850 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MTHADR_only.yaml seed=8272 name=momentum_gps_MTHADR_only_3_new trainer.max_steps=20000 trainer.devices=2

