# download TDC dataset
python preprocessing.py

seed_num=8272
# run multi task learning
python run.py experiment=encoder_train/gps/multitask/gps_MTHAD.yaml seed=${seed_num} 

 # run finetuning
root='./configs/experiment/encoder_train/gps/finetuning/template/*'
for d in $root; 
do
    echo $d
    python run.py experiment=$d seed=${seed_num}
done

# inference test
python run.py experiment=encoder_train/gps/inference/inference_test.yaml seed=${seed_num}

