
finetunemodel="FT0_mt10k_layernorm"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MT0_1_20k/2023-09-20/16-56-48/checkpoints/checkpoint_000014216.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MT0_2_20k/2023-09-20/16-56-48/checkpoints/checkpoint_000010931.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MT0_3_20k/2023-09-20/16-56-48/checkpoints/checkpoint_000012374.ckpt
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

finetunemodel="FTH_mt10k_layernorm"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTH_1_new/2023-09-20/16-56-48/checkpoints/checkpoint_000005447.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTH_2_new/2023-09-20/16-56-47/checkpoints/checkpoint_000018353.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTH_3_new/2023-09-20/16-56-47/checkpoints/checkpoint_000019583.ckpt
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

finetunemodel="FTHA_mt10k_layernorm"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTHA_1_new/2023-09-20/16-56-47/checkpoints/checkpoint_000018598.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTHA_2_new/2023-09-20/16-56-47/checkpoints/checkpoint_000018763.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTHA_3_new/2023-09-20/16-56-47/checkpoints/checkpoint_000016809.ckpt
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

finetunemodel="FTHAD_mt10k_layernorm"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTHAD_1_new/2023-09-20/16-56-47/checkpoints/checkpoint_000009813.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTHAD_2_new/2023-09-20/16-56-47/checkpoints/checkpoint_000018353.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTHAD_3_new/2023-09-20/16-56-47/checkpoints/checkpoint_000009552.ckpt
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

finetunemodel="FTHD_mt10k_layernorm"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTHD_1_new/2023-09-20/16-56-47/checkpoints/checkpoint_000005101.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTHD_2_new/2023-09-20/16-56-47/checkpoints/checkpoint_000019253.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTHD_3_new/2023-09-20/16-56-47/checkpoints/checkpoint_000014461.ckpt
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

finetunemodel="FTR_mt10k_layernorm"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTR_1_new/2023-09-20/16-56-47/checkpoints/checkpoint_000006384.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTR_2_new/2023-09-20/18-42-39/checkpoints/checkpoint_000018353.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTR_3_new/2023-09-20/18-44-10/checkpoints/checkpoint_000018875.ckpt
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

finetunemodel="FTR_only_mt10k_layernorm"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTR_only_1_new/2023-09-20/16-56-57/checkpoints/checkpoint_000008993.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTR_only_2_new/2023-09-20/16-56-57/checkpoints/checkpoint_000001821.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/momentum_gps_MTR_only_3_new/2023-09-20/16-56-58/checkpoints/checkpoint_000015398.ckpt
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