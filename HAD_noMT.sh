finetunemodel="FTHAD_gps_no_MT_epoch50"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_PTHAD/2023-09-27/11-22-37/checkpoints/checkpoint_000110264.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_PTHAD/2023-09-27/11-22-37/checkpoints/checkpoint_000110264.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_PTHAD/2023-09-27/11-22-37/checkpoints/checkpoint_000110264.ckpt
"
seed_num=8242
for check_path in $list_checkpoint;
do
    seed_num=$(($seed_num + 10))
    for d in /db2/users/jungwookim/projects/admet_prediction/configs/experiment/encoder_train/gps/finetuning/template_nomulti/*     # list directories in the form "/tmp/dirname/"
    do
        replace='/db2/users/jungwookim/projects/admet_prediction/configs/experiment/'
        replacewith=''
        new="${d/${replace}/${replacewith}}"
        #echo ${new}
        task=$(basename ${d} .yaml)
        echo ${task}
        echo ${seed_num}
        echo ${check_path}
        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32 -T dgx /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=${new} name=gps_${finetunemodel}_${seed_num}/${task} seed=${seed_num} system.checkpoint_path=${check_path} logger.wandb_logger.project=ADMET_${finetunemodel} trainer.max_epochs=50
    done
done