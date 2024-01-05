
finetunemodel="FT0_mt10k_new"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MT0_1/2023-08-17/17-41-38/checkpoints/checkpoint_000008519.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MT0_2/2023-08-17/17-41-38/checkpoints/checkpoint_000009568.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MT0_3/2023-08-17/17-41-38/checkpoints/checkpoint_000007481.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32 -T dgx /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=20
    done
done

finetunemodel="FTH_mt10k_new"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTH_1_new/2023-09-11/09-56-17/checkpoints/checkpoint_000013854.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTH_2_new/2023-09-11/09-56-17/checkpoints/checkpoint_000019285.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTH_3_new/2023-09-11/09-56-17/checkpoints/checkpoint_000019679.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32 -T dgx /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=20
    done
done

finetunemodel="FTHA_mt10k_new"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTHA_1_new/2023-09-11/09-56-17/checkpoints/checkpoint_000008125.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTHA_2_new/2023-09-11/09-56-17/checkpoints/checkpoint_000016234.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTHA_3_new/2023-09-11/09-56-17/checkpoints/checkpoint_000015920.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32 -T dgx /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=20
    done
done

finetunemodel="FTHAD_mt10k_new"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTHAD_1_new/2023-09-11/09-56-17/checkpoints/checkpoint_000017102.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTHAD_2_new/2023-09-11/09-56-17/checkpoints/checkpoint_000011112.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTHAD_3_new/2023-09-11/09-56-17/checkpoints/checkpoint_000015840.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32 -T dgx /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=20
    done
done

finetunemodel="FTHD_mt10k_new"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTHD_1_new/2023-09-11/09-56-17/checkpoints/checkpoint_000016543.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTHD_2_new/2023-09-11/09-56-17/checkpoints/checkpoint_000010633.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTHD_3_new/2023-09-11/09-56-17/checkpoints/checkpoint_000007305.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32 -T dgx /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=20
    done
done


finetunemodel="FTR_mt10k_new"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTR_1/2023-08-18/00-27-25/checkpoints/checkpoint_000008684.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTR_2/2023-08-18/00-28-31/checkpoints/checkpoint_000009094.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTR_3/2023-08-18/00-28-31/checkpoints/checkpoint_000009291.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32 -T dgx /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=20
    done
done

finetunemodel="FTHADR_mt10k_new"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTHADR_1/2023-08-19/11-31-00/checkpoints/checkpoint_000009254.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTHADR_2/2023-08-19/11-31-00/checkpoints/checkpoint_000008093.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTHADR_3/2023-08-19/11-31-00/checkpoints/checkpoint_000009685.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32 -T dgx /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=20
    done
done

finetunemodel="FTR_only_mt10k_new"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTR_1/2023-08-18/00-27-25/checkpoints/checkpoint_000008684.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTR_2/2023-08-18/00-28-31/checkpoints/checkpoint_000009094.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTR_3/2023-08-18/00-28-31/checkpoints/checkpoint_000009291.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32 -T dgx /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=20
    done
done

finetunemodel="FTHADR_only_mt10k_new"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTHADR_1/2023-08-19/11-31-00/checkpoints/checkpoint_000009254.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTHADR_2/2023-08-19/11-31-00/checkpoints/checkpoint_000008093.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTHADR_3/2023-08-19/11-31-00/checkpoints/checkpoint_000009685.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32 -T dgx /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=20
    done
done