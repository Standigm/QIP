
finetunemodel="FTR_only_mt10k_latest"
list_checkpoint="
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTR_only_1_new/2023-09-13/17-23-09/checkpoints/checkpoint_000004329.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTR_only_2_new/2023-09-13/17-24-09/checkpoints/checkpoint_000009472.ckpt
/db2/users/jungwookim/projects/admet_prediction/outputs/train/runs/gps_MTR_only_3_new/2023-09-13/17-27-06/checkpoints/checkpoint_000008290.ckpt
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