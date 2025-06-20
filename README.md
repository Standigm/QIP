# Quantum informed graph transformer model

Machine learning model that predict the ADMET (absorption, distribution, metabolism, excretion, and toxicity) properties of molecules.

## Environment setting & installation

```bash
mamba env create -f requirements.yml
mamba activate qip
pip install -e .
```

## How to run

Configuration is implemented depending on omegaconf and hydra package. 
You can refer to the contents of the corresponding package for instructions on how to use it.

## Download model weights

We provide pretrained model weights for the HAD-pretrained model, and multitask-trained/individually fine-tuned models for three independent experiments with different random seeds.

https://zenodo.org/records/15703563

Create saved_model directory and put weights into the saved_model directory.

## Model training
You can execute specific configuration through experiment argument.
```bash
# bash
bash workflow.sh

python run.py experiment=<your_config_to_run>

# download tdc datset
python preprocessing.py

# multitask learning
python run.py experiment=encoder_train/gps/multitask/gps_MTHAD.yaml seed=8272

# finetuning
python run.py experiment=encoder_train/gps/finetuning/template/ames.yaml seed=8272 
## example 
python run.py experiment=encoder_train/gps/finetuning/template/ames.yaml seed=8272 system.checkpoint_path=${model_dir}/multitask_weight_HAD.ckpt

# If you wanna print full error log, use HYDRA_FULL_ERROR=1 option
HYDRA_FULL_ERROR=1 python run.py experiment=encoder_train/gps/finetuning/template/ames.yaml seed=8272 system.checkpoint_path=${model_dir}/multitask_weight_HAD.ckpt
# inference
python run.py experiment=encoder_train/gps/inference/inference_test.yaml seed=8272
```


you can change specific argument by passing <arg_name>=<value>
```bash
python run.py experiment=<your_config_to_run> datamodule.batch_size=4 callbacks=early_stopping
```
HYDRA_FULL_ERROR=1 /db2/slurm_script/submit_job -N 1 -G 1 -C 32 -T dgx env MASTER_PORT=45100 /db2/users/hyunjunji/conda/envs/qip/bin/python run.py experiment=encoder_train/gps/finetuning/template/ames.yaml seed=8252 name=qipinferenc trainer.devices=1
/db2/users/hyunjunji/conda/envs/qip/bin/python

## Inference script

An inference script is available as well:

```
> python inference_script.py --help
usage: inference_script.py [-h] [--csv_path CSV_PATH] [--smiles_column SMILES_COLUMN] [--output_path OUTPUT_PATH] [--ckpt_path CKPT_PATH]
                           [--encoder_config ENCODER_CONFIG] [--task_head_config TASK_HEAD_CONFIG]

options:
  -h, --help            show this help message and exit
  --csv_path CSV_PATH   Path to csv file (default: None)
  --smiles_column SMILES_COLUMN
                        Column name of smiles (default: smiles)
  --output_path OUTPUT_PATH
                        Path to save result (default: qip_result.csv)
  --ckpt_path CKPT_PATH
                        Path to model checkpoint (default: saved_model/multitask_weight_HAD.ckpt)
  --encoder_config ENCODER_CONFIG
                        Path to encoder config (default: configs/system/encoder_config/gps/medium.yaml)
  --task_head_config TASK_HEAD_CONFIG
                        Path to task head config (default: configs/system/task_head_configs/gps/MT0.yaml)
```