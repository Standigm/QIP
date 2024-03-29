# download TDC dataset
python preprocessing.py

seed_num=8272
# run multi task learning
python run.py experiment=encoder_train/gps/multitask/gps_MTHAD.yaml seed=${seed_num} # trainer.max_steps=10 datamodule.batch_size=10

 # run finetuning
root='./configs/experiment/encoder_train/gps/finetuning/template/*'
for d in $root; 
do
    
    replace='./configs/experiment/'
    replacewith=''
    path="${d/${replace}/${replacewith}}"
    echo $path
    
    python run.py experiment=$path seed=${seed_num} # trainer.max_steps=10 datamodule.batch_size=10
done

# inference test
python run.py experiment=encoder_train/gps/inference/inference_test.yaml seed=${seed_num}

