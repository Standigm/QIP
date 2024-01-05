finetunemodel="FTHADRS_lr_t1_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTHADRS_1_new/2023-10-25/11-44-56/checkpoints/checkpoint_000007928.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTHADRS_2_new/2023-10-25/11-44-56/checkpoints/checkpoint_000009701.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTHADRS_3_new/2023-10-25/11-44-55/checkpoints/checkpoint_000008844.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32  /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=10
    done
done

finetunemodel="FTHADS_lr_t1_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTHADS_1_new/2023-10-25/11-44-55/checkpoints/checkpoint_000010239.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTHADS_2_new/2023-10-25/11-44-55/checkpoints/checkpoint_000016793.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTHADS_3_new/2023-10-25/11-44-55/checkpoints/checkpoint_000015675.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32  /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=10
    done
done


finetunemodel="FTHD_lr_t1_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTHD_1_new/2023-10-25/11-43-59/checkpoints/checkpoint_000010569.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTHD_2_new/2023-10-25/11-43-59/checkpoints/checkpoint_000019796.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTHD_3_new/2023-10-25/11-44-00/checkpoints/checkpoint_000007071.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32  /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=10
    done
done

finetunemodel="FTR_lr_t1_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTR_1_new/2023-10-25/11-43-59/checkpoints/checkpoint_000013950.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTR_2_new/2023-10-25/11-43-59/checkpoints/checkpoint_000015936.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTR_3_new/2023-10-25/11-43-59/checkpoints/checkpoint_000016037.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32  /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=10
    done
done


finetunemodel="FTR_only_lr_t1_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTR_only_1_new/2023-10-25/11-43-59/checkpoints/checkpoint_000018662.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTR_only_2_new/2023-10-25/11-43-59/checkpoints/checkpoint_000018353.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTR_only_3_new/2023-10-25/11-43-59/checkpoints/checkpoint_000011964.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32  /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=10
    done
done

finetunemodel="FTRS_lr_t1_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTRS_1_new/2023-10-25/11-44-55/checkpoints/checkpoint_000016037.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTRS_2_new/2023-10-25/11-44-54/checkpoints/checkpoint_000011080.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTRS_3_new/2023-10-25/11-44-54/checkpoints/checkpoint_000012981.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32  /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=10
    done
done

finetunemodel="FTS_lr_t1_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTS_1_new/2023-10-25/11-44-55/checkpoints/checkpoint_000009845.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTS_2_new/2023-10-25/11-44-55/checkpoints/checkpoint_000009435.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test1_eq_prev__gps_MTS_3_new/2023-10-25/11-44-55/checkpoints/checkpoint_000009648.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32  /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=10
    done
done