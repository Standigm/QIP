
finetunemodel="FTHADR_mt20k_momentum"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTHADR_1_new/2023-09-20/18-45-39/checkpoints/checkpoint_000008354.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTHADR_2_new/2023-09-21/06-21-48/checkpoints/checkpoint_000016596.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTHADR_3_new/2023-09-21/06-22-13/checkpoints/checkpoint_000019285.ckpt
"
seed_num=8242
for check_path in $list_checkpoint;
do
    seed_num=$(($seed_num + 10))
    for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template/*     # list directories in the form "/tmp/dirname/"
    do
        replace='/db2/users/jungwookim/projects/admet_prediction/configs/experiment/'
        replacewith=''
        new="${d/${replace}/${replacewith}}"
        #echo ${new}
        task=$(basename ${d} .yaml)
        echo ${task}
        echo ${seed_num}
        echo ${check_path}
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32  /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=20
    done
done

finetunemodel="FTHADR_only_mt20k_momentum"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTHADR_only_1_new/2023-09-20/16-56-57/checkpoints/checkpoint_000014248.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTHADR_only_2_new/2023-09-20/16-56-57/checkpoints/checkpoint_000006778.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTHADR_only_3_new/2023-09-20/16-56-59/checkpoints/checkpoint_000016724.ckpt
"
seed_num=8242
for check_path in $list_checkpoint;
do
    seed_num=$(($seed_num + 10))
    for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template/*     # list directories in the form "/tmp/dirname/"
    do
        replace='/db2/users/jungwookim/projects/admet_prediction/configs/experiment/'
        replacewith=''
        new="${d/${replace}/${replacewith}}"
        #echo ${new}
        task=$(basename ${d} .yaml)
        echo ${task}
        echo ${seed_num}
        echo ${check_path}
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32  /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=20
    done
done

finetunemodel="FTHADRS_mt20k_momentum"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTHADRS_1_new/2023-09-20/16-56-58/checkpoints/checkpoint_000008040.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTHADRS_2_new/2023-09-20/16-56-57/checkpoints/checkpoint_000017464.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTHADRS_3_new/2023-09-20/16-56-57/checkpoints/checkpoint_000005841.ckpt
"
seed_num=8242
for check_path in $list_checkpoint;
do
    seed_num=$(($seed_num + 10))
    for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template/*     # list directories in the form "/tmp/dirname/"
    do
        replace='/db2/users/jungwookim/projects/admet_prediction/configs/experiment/'
        replacewith=''
        new="${d/${replace}/${replacewith}}"
        #echo ${new}
        task=$(basename ${d} .yaml)
        echo ${task}
        echo ${seed_num}
        echo ${check_path}
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32  /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=20
    done
done

finetunemodel="FTRS_mt20k_momentum"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTRS_1_new/2023-09-20/16-56-56/checkpoints/checkpoint_000014493.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTRS_2_new/2023-09-20/16-56-56/checkpoints/checkpoint_000009632.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTRS_3_new/2023-09-20/16-56-58/checkpoints/checkpoint_000015675.ckpt
"
seed_num=8242
for check_path in $list_checkpoint;
do
    seed_num=$(($seed_num + 10))
    for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template/*     # list directories in the form "/tmp/dirname/"
    do
        replace='/db2/users/jungwookim/projects/admet_prediction/configs/experiment/'
        replacewith=''
        new="${d/${replace}/${replacewith}}"
        #echo ${new}
        task=$(basename ${d} .yaml)
        echo ${task}
        echo ${seed_num}
        echo ${check_path}
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32  /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=20
    done
done

finetunemodel="FTS_mt20k_momentum"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTS_1_new/2023-09-20/16-56-56/checkpoints/checkpoint_000003397.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTS_2_new/2023-09-20/16-57-03/checkpoints/checkpoint_000002380.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTS_3_new/2023-09-20/16-57-03/checkpoints/checkpoint_000016724.ckpt
"
seed_num=8242
for check_path in $list_checkpoint;
do
    seed_num=$(($seed_num + 10))
    for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template/*     # list directories in the form "/tmp/dirname/"
    do
        replace='/db2/users/jungwookim/projects/admet_prediction/configs/experiment/'
        replacewith=''
        new="${d/${replace}/${replacewith}}"
        #echo ${new}
        task=$(basename ${d} .yaml)
        echo ${task}
        echo ${seed_num}
        echo ${check_path}
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32  /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=20
    done
done