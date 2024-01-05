
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

finetunemodel="FTH_split_task_cls"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTH_1_new/2023-11-02/15-29-35/checkpoints/checkpoint_000008322.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTH_2_new/2023-11-02/15-29-35/checkpoints/checkpoint_000002721.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTH_3_new/2023-11-02/15-29-35/checkpoints/checkpoint_000002492.ckpt
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

finetunemodel="FTHA_split_task_cls"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTHA_1_new/2023-11-02/15-29-35/checkpoints/checkpoint_000001033.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTHA_2_new/2023-11-02/15-29-35/checkpoints/checkpoint_000001001.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTHA_3_new/2023-11-02/15-29-57/checkpoints/checkpoint_000003988.ckpt
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

finetunemodel="FTHAD_split_task_cls"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTHAD_1_new/2023-11-02/15-30-35/checkpoints/checkpoint_000006890.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTHAD_2_new/2023-11-02/15-31-37/checkpoints/checkpoint_000001246.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTHAD_3_new/2023-11-02/15-31-37/checkpoints/checkpoint_000002822.ckpt
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

finetunemodel="FTHADR_split_task_cls"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTHADR_1_new/2023-11-02/21-14-50/checkpoints/checkpoint_000000719.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTHADR_2_new/2023-11-02/21-36-53/checkpoints/checkpoint_000001608.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTHADR_3_new/2023-11-02/21-39-56/checkpoints/checkpoint_000000687.ckpt
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

finetunemodel="FTHADS_split_task_cls"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTHADS_1_new/2023-11-02/21-42-28/checkpoints/checkpoint_000000735.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTHADS_2_new/2023-11-02/21-43-53/checkpoints/checkpoint_000002476.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTHADS_3_new/2023-11-02/21-44-28/checkpoints/checkpoint_000005351.ckpt
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


finetunemodel="FTHD_split_task_cls"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTHD_1_new/2023-11-02/15-29-57/checkpoints/checkpoint_000000719.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTHD_2_new/2023-11-02/15-29-57/checkpoints/checkpoint_000005463.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTHD_3_new/2023-11-02/15-30-35/checkpoints/checkpoint_000001097.ckpt
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

finetunemodel="FTR_split_task_cls"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTR_1_new/2023-11-02/15-31-37/checkpoints/checkpoint_000001395.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTR_2_new/2023-11-02/15-45-23/checkpoints/checkpoint_000002098.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTR_3_new/2023-11-02/21-14-50/checkpoints/checkpoint_000002098.ckpt
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




finetunemodel="FTRS_split_task_cls"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTRS_1_new/2023-11-02/21-45-28/checkpoints/checkpoint_000001965.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTRS_2_new/2023-11-02/21-46-30/checkpoints/checkpoint_000000426.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTRS_3_new/2023-11-02/21-47-29/checkpoints/checkpoint_000002737.ckpt
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

finetunemodel="FTS_split_task_cls"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTS_1_new/2023-11-02/21-40-53/checkpoints/checkpoint_000003541.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTS_2_new/2023-11-02/21-41-52/checkpoints/checkpoint_000003972.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/split_classification_gps_MTS_3_new/2023-11-02/21-41-56/checkpoints/checkpoint_000000687.ckpt
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