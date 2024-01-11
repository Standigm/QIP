# Quantum informed graph transformer model
Machine learning model that predict the ADMET (absorption, distribution, metabolism, excretion, and toxicity) properties of molecules.

## Environment setting & installation
```bash
mamba env create -f requirements.yml
mamba activate qip
pip install -e .
```
#TODO: Symbolic link 걸기
# How to run
Configuration is implemented depending on omegaconf and hydra package. 
You can refer to the contents of the corresponding package for instructions on how to use it.

## train process
You can execute specific configuration through experiment argument.
```bash
python run.py experiment=<your_config_to_run>

# download tdc datset
python preprocessing.py

# multitask learning
python run.py experiment=encoder_train/gps/multitask/gps_MTHAD.yaml seed=8272

# finetuning
python run.py experiment=encoder_train/gps/finetuning/template/ames.yaml seed=8272 system.checkpoint_path={saved_multitask_weight.ckpt}
## example 
python run.py experiment=encoder_train/gps/finetuning/template/ames.yaml seed=8272 system.checkpoint_path=${model_dir}/multitask_weight_HAD.ckpt


HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32 -T dgx env MASTER_PORT=45100 /db2/users/jungwookim/mambaforge-pypy3/envs/grpe/bin/python run.py experiment=encoder_train/gps/finetuning/template/ames.yaml seed=8272 system.checkpoint_path=./saved_model/multitask_weight_HAD.ckpt
# inference

python run.py experiment=encoder_train/gps/inference/inference_test.yaml seed=8272
```


you can change specific argument by passing <arg_name>=<value>
```bash
python run.py experiment=<your_config_to_run> datamodule.batch_size=4 callbacks=early_stopping
```
