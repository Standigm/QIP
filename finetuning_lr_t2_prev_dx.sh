


finetunemodel="FT0_lr_t2_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MT0_1_new/2023-10-27/10-37-23/checkpoints/checkpoint_000006480.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MT0_2_new/2023-10-27/11-07-01/checkpoints/checkpoint_000017762.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MT0_3_new/2023-10-27/11-32-53/checkpoints/checkpoint_000007321.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32 -T dgx /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=10
    done
done

finetunemodel="FTH_lr_t2_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTH_1_new/2023-10-27/11-32-54/checkpoints/checkpoint_000004792.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTH_2_new/2023-10-27/11-39-58/checkpoints/checkpoint_000009222.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTH_3_new/2023-10-27/11-40-29/checkpoints/checkpoint_000015808.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32 -T dgx /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=10
    done
done

finetunemodel="FTHA_lr_t2_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTHA_1_new/2023-10-27/11-42-37/checkpoints/checkpoint_000009552.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTHA_2_new/2023-10-27/11-43-44/checkpoints/checkpoint_000007566.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTHA_3_new/2023-10-27/11-44-51/checkpoints/checkpoint_000013215.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32 -T dgx /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=10
    done
done

finetunemodel="FTHAD_lr_t2_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTHAD_1_new/2023-10-27/12-14-03/checkpoints/checkpoint_000014051.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTHAD_2_new/2023-10-27/12-21-07/checkpoints/checkpoint_000003232.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTHAD_3_new/2023-10-27/12-25-58/checkpoints/checkpoint_000010489.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32 -T dgx /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=10
    done
done

finetunemodel="FTHADR_lr_t2_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTHADR_1_new/2023-10-27/10-08-29/checkpoints/checkpoint_000008040.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTHADR_2_new/2023-10-27/10-08-29/checkpoints/checkpoint_000005612.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTHADR_3_new/2023-10-27/10-08-29/checkpoints/checkpoint_000004169.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32 -T dgx /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=10
    done
done


finetunemodel="FTHADR_only_lr_t2_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTHADR_only_1_new/2023-10-27/10-08-29/checkpoints/checkpoint_000004941.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTHADR_only_2_new/2023-10-27/10-08-29/checkpoints/checkpoint_000010750.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTHADR_only_3_new/2023-10-27/10-08-29/checkpoints/checkpoint_000010601.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32 -T dgx  /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=10
    done
done