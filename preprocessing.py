from tdc import utils
from tdc.benchmark_group import admet_group
import os
import pandas as pd


def download_tdc():
    names = utils.retrieve_benchmark_names('ADMET_Group')
    group = admet_group(path='./datasets/')
    
    admet_groups = os.listdir('./datasets/admet_group')
    admet_groups.remove('.DS_Store')
    for admet in admet_groups:
        os.makedirs(f'./datasets/admet_group/{admet}/raw/', exist_ok=True)
        train = pd.read_csv(f'./datasets/admet_group/{admet}/train_val.csv')
        train = train.rename(columns={'Drug':'smiles', 'Y': 'labels'})
        train['smiles'].to_csv(f'./datasets/admet_group/{admet}/raw/trainset_smiles.csv',index=False)
        train['labels'].to_csv(f'./datasets/admet_group/{admet}/raw/trainset_labels.csv',index=False)
        
        test = pd.read_csv(f'./datasets/admet_group/{admet}/test.csv')    
        test = test.rename(columns={'Drug':'smiles', 'Y': 'labels'})
        test['smiles'].to_csv(f'./datasets/admet_group/{admet}/raw/testset_smiles.csv',index=False)
        test['labels'].to_csv(f'./datasets/admet_group/{admet}/raw/testset_labels.csv',index=False)
        
if __name__=="__main__":
    download_tdc()