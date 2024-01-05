
# finetunemodel="gatv2_MT0"
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


num_layer="15 20"

d_model="384 512"

n_head="16"

port_num=46000
for n_layer in ${num_layer};
do
    for dimension in ${d_model};
    do
        for head in ${n_head};
        do
            
            port_num=$(($port_num + 10))
            echo ${port_num}

            HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=${port_num} /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MT0_gnn.yaml name=gatv2_${n_layer}_${dimension}_${head} seed=8252 system.encoder_config.module.d_model=${dimension} system.encoder_config.module.nhead=${head} system.encoder_config.module.num_layer=${n_layer}
        done
    done
done        
   
for n_layer in ${num_layer};
do
    for dimension in ${d_model};
    do
        port_num=$(($port_num + 10))
        echo ${port_num}

        HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 2 -C 32  env MASTER_PORT=${port_num} /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/multitask/gps_MT0_gine.yaml name=gine_${n_layer}_${dimension}_${head} seed=8252 system.encoder_config.module.d_model=${dimension} system.encoder_config.module.nhead=${head} system.encoder_config.module.num_layer=${n_layer}
   
    done
done        