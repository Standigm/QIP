
finetunemodel="FT0_lr_equal_prev_ft_lrepoch"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MT0_1_new/2023-10-22/21-58-44/checkpoints/checkpoint_000007087.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MT0_2_new/2023-10-22/21-58-44/checkpoints/checkpoint_000002593.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MT0_3_new/2023-10-22/21-58-44/checkpoints/checkpoint_000008945.ckpt
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

finetunemodel="FTH_lr_equal_prev_ft_lrepoch"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTH_1_new/2023-10-22/22-12-59/checkpoints/checkpoint_000004824.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTH_2_new/2023-10-22/21-58-44/checkpoints/checkpoint_000004526.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTH_3_new/2023-10-22/21-58-44/checkpoints/checkpoint_000004494.ckpt
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

finetunemodel="FTHA_lr_equal_prev_ft_lrepoch"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTHA_1_new/2023-10-22/21-58-44/checkpoints/checkpoint_000009978.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTHA_2_new/2023-10-22/21-58-44/checkpoints/checkpoint_000004068.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTHA_3_new/2023-10-22/21-58-44/checkpoints/checkpoint_000004723.ckpt
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

finetunemodel="FTHAD_lr_equal_prev_ft_lrepoch"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTHAD_1_new/2023-10-22/21-58-45/checkpoints/checkpoint_000006432.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTHAD_2_new/2023-10-22/21-58-45/checkpoints/checkpoint_000005367.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTHAD_3_new/2023-10-22/21-58-45/checkpoints/checkpoint_000009307.ckpt
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

finetunemodel="FTHADR_lr_equal_prev_ft_lrepoch"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTHADR_1_new/2023-10-22/21-58-46/checkpoints/checkpoint_000003956.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTHADR_2_new/2023-10-22/21-58-47/checkpoints/checkpoint_000005170.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTHADR_3_new/2023-10-22/21-58-47/checkpoints/checkpoint_000001768.ckpt
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


finetunemodel="FTHADR_only_lr_equal_prev_ft_lrepoch"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTHADR_only_1_new/2023-10-22/21-58-47/checkpoints/checkpoint_000003168.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTHADR_only_2_new/2023-10-22/21-58-47/checkpoints/checkpoint_000003541.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/equal_prev_10k_gps_MTHADR_only_3_new/2023-10-22/21-58-46/checkpoints/checkpoint_000005447.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32 -T dgx  /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=20
    done
done

