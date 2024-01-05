finetunemodel="FTHADRS_lr_equal_prev_ft_lrepoch"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTHADRS_1_new/2023-10-22/21-58-48/checkpoints/checkpoint_000002806.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTHADRS_2_new/2023-10-23/08-46-57/checkpoints/checkpoint_000001230.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTHADRS_3_new/2023-10-23/08-47-45/checkpoints/checkpoint_000001262.ckpt
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

finetunemodel="FTHADS_lr_equal_prev_ft_lrepoch"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTHADS_1_new/2023-10-22/21-58-46/checkpoints/checkpoint_000001768.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTHADS_2_new/2023-10-22/21-58-46/checkpoints/checkpoint_000009765.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTHADS_3_new/2023-10-22/21-58-46/checkpoints/checkpoint_000004036.ckpt
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


finetunemodel="FTHD_lr_equal_prev_ft_lrepoch"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTHD_1_new/2023-10-22/21-58-44/checkpoints/checkpoint_000001789.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTHD_2_new/2023-10-22/21-58-44/checkpoints/checkpoint_000009403.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTHD_3_new/2023-10-22/21-58-45/checkpoints/checkpoint_000008583.ckpt
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

finetunemodel="FTR_lr_equal_prev_ft_lrepoch"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTR_1_new/2023-10-22/21-58-44/checkpoints/checkpoint_000003200.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTR_2_new/2023-10-22/21-58-44/checkpoints/checkpoint_000001459.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTR_3_new/2023-10-22/21-58-44/checkpoints/checkpoint_000006123.ckpt
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


finetunemodel="FTR_only_lr_equal_prev_ft_lrepoch"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTR_only_1_new/2023-10-22/21-58-47/checkpoints/checkpoint_000006432.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTR_only_2_new/2023-10-22/21-58-49/checkpoints/checkpoint_000005548.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTR_only_3_new/2023-10-22/21-58-46/checkpoints/checkpoint_000006384.ckpt
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

finetunemodel="FTRS_lr_equal_prev_ft_lrepoch"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTRS_1_new/2023-10-22/21-58-47/checkpoints/checkpoint_000001901.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTRS_2_new/2023-10-22/21-58-47/checkpoints/checkpoint_000007156.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTRS_3_new/2023-10-22/21-58-48/checkpoints/checkpoint_000006911.ckpt
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

finetunemodel="FTS_lr_equal_prev_ft_lrepoch"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTS_1_new/2023-10-22/21-58-47/checkpoints/checkpoint_000002774.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTS_2_new/2023-10-22/21-58-47/checkpoints/checkpoint_000007156.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTS_3_new/2023-10-22/21-58-46/checkpoints/checkpoint_000009600.ckpt
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