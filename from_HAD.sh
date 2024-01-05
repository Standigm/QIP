finetunemodel="FT_No_multitask_HAD"

list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MT0_1/2023-08-17/17-41-38/checkpoints/checkpoint_000008519.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MT0_2/2023-08-17/17-41-38/checkpoints/checkpoint_000009568.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MT0_3/2023-08-17/17-41-38/checkpoints/checkpoint_000007481.ckpt
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
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32 -T dgx /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_PTHAD/2023-08-03/09-43-46/checkpoints/checkpoint_000122420.ckpt logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=20
    done
done