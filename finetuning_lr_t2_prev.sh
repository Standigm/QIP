finetunemodel="FTHADRS_lr_t2_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTHADRS_1_new/2023-10-27/10-08-29/checkpoints/checkpoint_000005463.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTHADRS_2_new/2023-10-27/10-08-29/checkpoints/checkpoint_000002673.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTHADRS_3_new/2023-10-27/10-08-29/checkpoints/checkpoint_000002774.ckpt
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

finetunemodel="FTHADS_lr_t2_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTHADS_1_new/2023-10-27/10-08-29/checkpoints/checkpoint_000011341.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTHADS_2_new/2023-10-27/10-08-29/checkpoints/checkpoint_000019780.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTHADS_3_new/2023-10-27/10-08-29/checkpoints/checkpoint_000003903.ckpt
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


finetunemodel="FTHD_lr_t2_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTHD_1_new/2023-10-27/11-45-57/checkpoints/checkpoint_000007023.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTHD_2_new/2023-10-27/11-53-29/checkpoints/checkpoint_000010143.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTHD_3_new/2023-10-27/12-13-30/checkpoints/checkpoint_000008945.ckpt
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

finetunemodel="FTR_lr_t2_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTR_1_new/2023-10-27/12-25-58/checkpoints/checkpoint_000014264.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTR_2_new/2023-10-27/12-27-05/checkpoints/checkpoint_000005415.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTR_3_new/2023-10-27/13-03-19/checkpoints/checkpoint_000005138.ckpt
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


finetunemodel="FTR_only_lr_t2_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTR_only_1_new/2023-10-27/13-19-35/checkpoints/checkpoint_000012310.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTR_only_2_new/2023-10-27/13-33-07/checkpoints/checkpoint_000005335.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTR_only_3_new/2023-10-27/15-39-46/checkpoints/checkpoint_000002902.ckpt
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

finetunemodel="FTRS_lr_t2_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTRS_1_new/2023-10-27/10-08-29/checkpoints/checkpoint_000001768.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTRS_2_new/2023-10-27/10-08-29/checkpoints/checkpoint_000005596.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTRS_3_new/2023-10-27/10-08-29/checkpoints/checkpoint_000002428.ckpt
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

finetunemodel="FTS_lr_t2_finetuning"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTS_1_new/2023-10-27/10-08-29/checkpoints/checkpoint_000003328.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTS_2_new/2023-10-27/10-08-29/checkpoints/checkpoint_000011570.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/lr_test2_eq_prev__gps_MTS_3_new/2023-10-27/10-08-29/checkpoints/checkpoint_000008812.ckpt
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