


finetunemodel="FT0_lr_t1_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MT0_1_new/2023-10-25/11-43-59/checkpoints/checkpoint_000018385.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MT0_2_new/2023-10-25/11-43-59/checkpoints/checkpoint_000019301.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MT0_3_new/2023-10-25/11-43-59/checkpoints/checkpoint_000019812.ckpt
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

finetunemodel="FTH_lr_t1_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTH_1_new/2023-10-25/11-43-59/checkpoints/checkpoint_000013130.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTH_2_new/2023-10-25/11-43-59/checkpoints/checkpoint_000017842.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTH_3_new/2023-10-25/11-43-59/checkpoints/checkpoint_000008615.ckpt
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

finetunemodel="FTHA_lr_t1_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTHA_1_new/2023-10-25/11-43-59/checkpoints/checkpoint_000009174.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTHA_2_new/2023-10-25/11-43-59/checkpoints/checkpoint_000017890.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTHA_3_new/2023-10-25/11-43-59/checkpoints/checkpoint_000011703.ckpt
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

finetunemodel="FTHAD_lr_t1_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTHAD_1_new/2023-10-25/11-44-00/checkpoints/checkpoint_000015792.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTHAD_2_new/2023-10-25/11-44-00/checkpoints/checkpoint_000009206.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTHAD_3_new/2023-10-25/11-44-00/checkpoints/checkpoint_000009110.ckpt
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

finetunemodel="FTHADR_lr_t1_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTHADR_1_new/2023-10-25/11-45-05/checkpoints/checkpoint_000009962.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTHADR_2_new/2023-10-25/11-45-07/checkpoints/checkpoint_000009451.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTHADR_3_new/2023-10-25/11-45-07/checkpoints/checkpoint_000015116.ckpt
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


finetunemodel="FTHADR_only_lr_t1_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTHADR_only_1_new/2023-10-25/11-44-55/checkpoints/checkpoint_000010569.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTHADR_only_2_new/2023-10-25/11-44-57/checkpoints/checkpoint_000009451.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTHADR_only_3_new/2023-10-25/11-44-55/checkpoints/checkpoint_000008844.ckpt
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