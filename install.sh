conda create -n qip python=3.10
conda activate qip
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install -c pyg pyg
mamba install -c conda-forge lightning python-dotenv hydra-core hydra-colorlog rdkit rich
mamba install -c openeye openeye-toolkits
mamba install -c pyg pytorch-scatter
pip install -e .
