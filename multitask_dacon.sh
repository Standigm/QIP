HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 env MASTER_PORT=47890 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/dacon/gps_dacon_addition.yaml seed=8252 logger.wandb_logger.project=Dacon name=gps_addition_HADR_1
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 env MASTER_PORT=47900 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/dacon/gps_dacon_addition.yaml seed=8262 logger.wandb_logger.project=Dacon name=gps_addition_HADR_2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 env MASTER_PORT=47910 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/dacon/gps_dacon_addition.yaml seed=8272 logger.wandb_logger.project=Dacon name=gps_addition_HADR_3

HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 env MASTER_PORT=47890 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/dacon/gps_dacon_addition.yaml seed=8252 logger.wandb_logger.project=Dacon
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 env MASTER_PORT=47900 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/dacon/gps_dacon_addition_0913.yaml seed=8252 logger.wandb_logger.project=Dacon

HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 env MASTER_PORT=47910 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/dacon/gps_dacon_addition_0913_only.yaml seed=8252 logger.wandb_logger.project=Dacon






HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 env MASTER_PORT=47910 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/dacon/gps_dacon_addtasks_pruned.yaml seed=8252 logger.wandb_logger.project=Dacon name=gps_Dacon_MT_HADR_pruned_addtask_1
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 env MASTER_PORT=47920 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/dacon/gps_dacon_addtasks_pruned.yaml seed=8262 logger.wandb_logger.project=Dacon name=gps_Dacon_MT_HADR_pruned_addtask_2
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 env MASTER_PORT=47930 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/dacon/gps_dacon_addtasks_pruned.yaml seed=8272 logger.wandb_logger.project=Dacon name=gps_Dacon_MT_HADR_pruned_addtask_3
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 env MASTER_PORT=47940 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/dacon/gps_dacon_addtasks_pruned.yaml seed=8282 logger.wandb_logger.project=Dacon name=gps_Dacon_MT_HADR_pruned_addtask_4
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 env MASTER_PORT=47950 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/dacon/gps_dacon_addtasks_pruned.yaml seed=8292 logger.wandb_logger.project=Dacon name=gps_Dacon_MT_HADR_pruned_addtask_5

HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 env MASTER_PORT=47910 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/dacon/gps_dacon_addtasks_pruned.yaml seed=8302 logger.wandb_logger.project=Dacon name=gps_Dacon_MT_HADR_pruned_addtask_6
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 env MASTER_PORT=47920 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/dacon/gps_dacon_addtasks_pruned.yaml seed=8312 logger.wandb_logger.project=Dacon name=gps_Dacon_MT_HADR_pruned_addtask_7
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 env MASTER_PORT=47930 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/dacon/gps_dacon_addtasks_pruned.yaml seed=8322 logger.wandb_logger.project=Dacon name=gps_Dacon_MT_HADR_pruned_addtask_8
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 env MASTER_PORT=47940 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/dacon/gps_dacon_addtasks_pruned.yaml seed=8332 logger.wandb_logger.project=Dacon name=gps_Dacon_MT_HADR_pruned_addtask_9
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32 env MASTER_PORT=47950 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/dacon/gps_dacon_addtasks_pruned.yaml seed=8342 logger.wandb_logger.project=Dacon name=gps_Dacon_MT_HADR_pruned_addtask_10
