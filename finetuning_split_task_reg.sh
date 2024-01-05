
finetunemodel="FT0_split_task_reg"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MT0_1_new/2023-11-03/10-13-41/checkpoints/checkpoint_000002764.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MT0_2_new/2023-11-03/10-14-15/checkpoints/checkpoint_000008346.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MT0_3_new/2023-11-03/10-14-40/checkpoints/checkpoint_000008030.ckpt
"
seed_num=8242
for check_path in $list_checkpoint;
do
    seed_num=$(($seed_num + 10))
    for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template_reg/*     # list directories in the form "/tmp/dirname/"
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

finetunemodel="FTH_split_task_reg"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTH_1_new/2023-11-02/15-28-45/checkpoints/checkpoint_000007414.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTH_2_new/2023-11-02/15-28-47/checkpoints/checkpoint_000007778.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTH_3_new/2023-11-02/15-28-48/checkpoints/checkpoint_000009198.ckpt
"
seed_num=8242
for check_path in $list_checkpoint;
do
    seed_num=$(($seed_num + 10))
    for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template_reg/*     # list directories in the form "/tmp/dirname/"
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

finetunemodel="FTHA_split_task_reg"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTHA_1_new/2023-11-02/15-29-54/checkpoints/checkpoint_000002898.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTHA_2_new/2023-11-02/15-29-56/checkpoints/checkpoint_000007644.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTHA_3_new/2023-11-02/15-32-05/checkpoints/checkpoint_000006680.ckpt
"
seed_num=8242
for check_path in $list_checkpoint;
do
    seed_num=$(($seed_num + 10))
    for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template_reg/*     # list directories in the form "/tmp/dirname/"
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

finetunemodel="FTHAD_split_task_reg"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTHAD_1_new/2023-11-02/15-38-40/checkpoints/checkpoint_000007044.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTHAD_2_new/2023-11-02/15-38-41/checkpoints/checkpoint_000001778.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTHAD_3_new/2023-11-02/15-40-44/checkpoints/checkpoint_000008764.ckpt
"
seed_num=8242
for check_path in $list_checkpoint;
do
    seed_num=$(($seed_num + 10))
    for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template_reg/*     # list directories in the form "/tmp/dirname/"
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

finetunemodel="FTHADR_split_task_reg"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTHADR_1_new/2023-11-02/15-51-34/checkpoints/checkpoint_000007044.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTHADR_2_new/2023-11-02/15-51-34/checkpoints/checkpoint_000003048.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTHADR_3_new/2023-11-02/15-53-13/checkpoints/checkpoint_000009364.ckpt
"
seed_num=8242
for check_path in $list_checkpoint;
do
    seed_num=$(($seed_num + 10))
    for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template_reg/*     # list directories in the form "/tmp/dirname/"
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



finetunemodel="FTHADRS_split_task_reg"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTHADRS_1_new/2023-11-03/10-13-00/checkpoints/checkpoint_000006766.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTHADRS_2_new/2023-11-03/10-13-02/checkpoints/checkpoint_000009664.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTHADRS_3_new/2023-11-03/10-13-07/checkpoints/checkpoint_000009980.ckpt
"
seed_num=8242
for check_path in $list_checkpoint;
do
    seed_num=$(($seed_num + 10))
    for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template_reg/*     # list directories in the form "/tmp/dirname/"
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

finetunemodel="FTHADS_split_task_reg"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTHADS_1_new/2023-11-02/16-01-19/checkpoints/checkpoint_000008416.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTHADS_2_new/2023-11-02/16-06-45/checkpoints/checkpoint_000008914.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTHADS_3_new/2023-11-02/16-07-51/checkpoints/checkpoint_000006064.ckpt
"
seed_num=8242
for check_path in $list_checkpoint;
do
    seed_num=$(($seed_num + 10))
    for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template_reg/*     # list directories in the form "/tmp/dirname/"
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


finetunemodel="FTHD_split_task_reg"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTHD_1_new/2023-11-02/15-34-49/checkpoints/checkpoint_000007414.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTHD_2_new/2023-11-02/15-35-50/checkpoints/checkpoint_000002994.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTHD_3_new/2023-11-02/15-36-25/checkpoints/checkpoint_000006632.ckpt
"
seed_num=8242
for check_path in $list_checkpoint;
do
    seed_num=$(($seed_num + 10))
    for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template_reg/*     # list directories in the form "/tmp/dirname/"
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

finetunemodel="FTR_split_task_reg"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTR_1_new/2023-11-02/15-48-24/checkpoints/checkpoint_000004382.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTR_2_new/2023-11-02/15-49-26/checkpoints/checkpoint_000003262.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTR_3_new/2023-11-02/15-48-51/checkpoints/checkpoint_000008598.ckpt
"
seed_num=8242
for check_path in $list_checkpoint;
do
    seed_num=$(($seed_num + 10))
    for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template_reg/*     # list directories in the form "/tmp/dirname/"
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




finetunemodel="FTRS_split_task_reg"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTRS_1_new/2023-11-02/16-08-23/checkpoints/checkpoint_000006482.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTRS_2_new/2023-11-02/16-08-54/checkpoints/checkpoint_000001698.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTRS_3_new/2023-11-02/20-14-17/checkpoints/checkpoint_000008694.ckpt
"
seed_num=8242
for check_path in $list_checkpoint;
do
    seed_num=$(($seed_num + 10))
    for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template_reg/*     # list directories in the form "/tmp/dirname/"
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

finetunemodel="FTS_split_task_reg"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTS_1_new/2023-11-02/15-53-13/checkpoints/checkpoint_000004698.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTS_2_new/2023-11-02/15-58-05/checkpoints/checkpoint_000007848.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_regression_gps_MTS_3_new/2023-11-02/16-01-19/checkpoints/checkpoint_000005212.ckpt
"
seed_num=8242
for check_path in $list_checkpoint;
do
    seed_num=$(($seed_num + 10))
    for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template_reg/*     # list directories in the form "/tmp/dirname/"
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