# finetunemodel="gatv2_FTHAD_paper"
# list_checkpoint="
# /db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gatv2_MTHAD_1/2023-11-19/12-26-40/checkpoints/checkpoint_000006219.ckpt
# /db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gatv2_MTHAD_2/2023-11-19/12-26-40/checkpoints/checkpoint_000004350.ckpt
# /db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gatv2_MTHAD_3/2023-11-19/12-26-40/checkpoints/checkpoint_000003823.ckpt
# "
# seed_num=8242
# for check_path in $list_checkpoint;
# do
#     seed_num=$(($seed_num + 10))
#     for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template_gatv2/*     # list directories in the form "/tmp/dirname/"
#     do
#         replace='/db2/users/jungwookim/projects/admet_prediction/configs/experiment/'
#         replacewith=''
#         new="${d/${replace}/${replacewith}}"
#         #echo ${new}
#         task=$(basename ${d} .yaml)
#         echo ${task}
#         echo ${seed_num}
#         echo ${check_path}
#         HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32  /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gatv2_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=20
#     done
# done

# finetunemodel="gine_FTHAD_paper"
# list_checkpoint="
# /db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gine_MTHAD_1/2023-11-18/15-41-35/checkpoints/checkpoint_000005708.ckpt
# /db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gine_MTHAD_2/2023-11-18/15-41-35/checkpoints/checkpoint_000005548.ckpt
# /db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gine_MTHAD_3/2023-11-18/15-41-36/checkpoints/checkpoint_000005367.ckpt
# "
# seed_num=8242
# for check_path in $list_checkpoint;
# do
#     seed_num=$(($seed_num + 10))
#     for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template_gine/*     # list directories in the form "/tmp/dirname/"
#     do
#         replace='/db2/users/jungwookim/projects/admet_prediction/configs/experiment/'
#         replacewith=''
#         new="${d/${replace}/${replacewith}}"
#         #echo ${new}
#         task=$(basename ${d} .yaml)
#         echo ${task}
#         echo ${seed_num}
#         echo ${check_path}
#         HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32  /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gine_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=20
#     done
# done

finetunemodel="gatv2_FT0_paper"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gatv2_MT0_1/2023-11-20/11-51-26/checkpoints/checkpoint_000004941.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gatv2_MT0_2/2023-11-20/11-51-26/checkpoints/checkpoint_000004329.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gatv2_MT0_3/2023-11-20/11-51-26/checkpoints/checkpoint_000006677.ckpt
"
seed_num=8242
for check_path in $list_checkpoint;
do
    seed_num=$(($seed_num + 10))
    for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template_gatv2/*     # list directories in the form "/tmp/dirname/"
    do
        replace='/db2/users/jungwookim/projects/admet_prediction/configs/experiment/'
        replacewith=''
        new="${d/${replace}/${replacewith}}"
        #echo ${new}
        task=$(basename ${d} .yaml)
        echo ${task}
        echo ${seed_num}
        echo ${check_path}
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32  /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gatv2_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=20
    done
done

finetunemodel="gine_FT0_paper"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gine_MT0_1/2023-11-20/11-51-26/checkpoints/checkpoint_000003706.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gine_MT0_2/2023-11-20/11-51-26/checkpoints/checkpoint_000006842.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gine_MT0_3/2023-11-20/11-51-26/checkpoints/checkpoint_000003562.ckpt
"
seed_num=8242
for check_path in $list_checkpoint;
do
    seed_num=$(($seed_num + 10))
    for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template_gine/*     # list directories in the form "/tmp/dirname/"
    do
        replace='/db2/users/jungwookim/projects/admet_prediction/configs/experiment/'
        replacewith=''
        new="${d/${replace}/${replacewith}}"
        #echo ${new}
        task=$(basename ${d} .yaml)
        echo ${task}
        echo ${seed_num}
        echo ${check_path}
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32  /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gine_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=20
    done
done
