# ADMET Prediction
Machine learning model that predict the ADMET (absorption, distribution, metabolism, excretion, and toxicity) properties of molecules.

## Environment setting & installation
```bash
mamba env create -f requirements.yml
mamba activate chemenv
pip install -e .
```
## Hook setting
```bash
pre-commit install
```

# How to run
Configuration is implemented depending on omegaconf and hydra package. 
You can refer to the contents of the corresponding package for instructions on how to use it.

## train process
You can execute specific configuration through experiment argument.
```bash
python run.py experiment=<your_config_to_run>
# example
python run.py experiment=encoder_train/grpe/CYP+hERG+lipo+MS+perm+sol+nabla/initial
```


you can change specific argument by passing <arg_name>=<value>
```bash
python run.py experiment=<your_config_to_run> datamodule.batch_size=4 callbacks=early_stopping
```
