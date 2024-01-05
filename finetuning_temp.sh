
# finetunemodel="FT0_split_task_reg"
# list_checkpoint="
# /db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MT0_1_new/2023-11-03/10-13-41/checkpoints/checkpoint_000002764.ckpt
# /db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MT0_2_new/2023-11-03/10-14-15/checkpoints/checkpoint_000008346.ckpt
# /db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MT0_3_new/2023-11-03/10-14-40/checkpoints/checkpoint_000008030.ckpt
# "
# seed_num=8242
# for check_path in $list_checkpoint;
# do
#     seed_num=$(($seed_num + 10))
#     for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template_reg/*     # list directories in the form "/tmp/dirname/"
#     do
#         replace='/db2/users/jungwookim/projects/admet_prediction/configs/experiment/'
#         replacewith=''
#         new="${d/${replace}/${replacewith}}"
#         #echo ${new}
#         task=$(basename ${d} .yaml)
#         echo ${task}
#         echo ${seed_num}
#         echo ${check_path}
#         HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32 -T dgx /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=20
#     done
# done


# finetunemodel="FTHADRS_split_task_reg"
# list_checkpoint="
# /db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTHADRS_1_new/2023-11-03/10-13-00/checkpoints/checkpoint_000006766.ckpt
# /db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTHADRS_2_new/2023-11-03/10-13-02/checkpoints/checkpoint_000009664.ckpt
# /db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTHADRS_3_new/2023-11-03/10-13-07/checkpoints/checkpoint_000009980.ckpt
# "
# seed_num=8242
# for check_path in $list_checkpoint;
# do
#     seed_num=$(($seed_num + 10))
#     for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template_reg/*     # list directories in the form "/tmp/dirname/"
#     do
#         replace='/db2/users/jungwookim/projects/admet_prediction/configs/experiment/'
#         replacewith=''
#         new="${d/${replace}/${replacewith}}"
#         #echo ${new}
#         task=$(basename ${d} .yaml)
#         echo ${task}
#         echo ${seed_num}
#         echo ${check_path}
#         HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32 -T dgx /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=20
#     done
# done


finetunemodel="FT0_split_task_cls"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MT0_1_new/2023-11-04/01-37-41/checkpoints/checkpoint_000003919.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MT0_2_new/2023-11-04/01-37-41/checkpoints/checkpoint_000002034.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MT0_3_new/2023-11-04/01-37-41/checkpoints/checkpoint_000004579.ckpt
"
seed_num=8242
for check_path in $list_checkpoint;
do
    seed_num=$(($seed_num + 10))
    for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template_cls/*     # list directories in the form "/tmp/dirname/"
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


finetunemodel="FTHADRS_split_task_cls"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTHADRS_1_new/2023-11-04/01-37-41/checkpoints/checkpoint_000000389.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTHADRS_2_new/2023-11-04/01-37-43/checkpoints/checkpoint_000001752.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTHADRS_3_new/2023-11-04/01-37-43/checkpoints/checkpoint_000000474.ckpt
"
seed_num=8242
for check_path in $list_checkpoint;
do
    seed_num=$(($seed_num + 10))
    for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template_cls/*     # list directories in the form "/tmp/dirname/"
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
