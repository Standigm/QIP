from pathlib import Path
from collections.abc import Callable

import hydra
import pandas as pd
import torch
from omegaconf import OmegaConf
from qip.datamodules.collaters import DefaultCollater
from qip.datamodules.featurizers import QIPFeaturizer
from qip.datamodules.transforms import RandomWalkGenerator
from rdkit import Chem
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch import nn


# inference
class InferenceModule:
    def __init__(
        self,
        encoder: nn.Module,
        task_heads: dict[str, nn.Module],
        task_head_post_processes: dict[str, Callable | None],
        device: torch.device,
    ) -> None:
        self.encoder = encoder
        self.task_heads = task_heads
        self.task_head_post_processes = task_head_post_processes
        self.device = device

    def __call__(self, batch: Data) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            batch = batch.to(self.device)
            encoder_out = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.rwse)
            task_outputs = {}
            for task_name, task_head in self.task_heads.items():
                task_output = task_head(encoder_out[0], encoder_out[1])
                postprocessor = self.task_head_post_processes.get(task_name, None)
                if postprocessor is not None:
                    task_output = postprocessor(task_output)
                task_outputs.update({task_name: task_output})
        return task_outputs


def _smiles_to_pyg_data(smiles: str) -> Data:
    featurizer = QIPFeaturizer()
    rw = RandomWalkGenerator(ksteps=[1, 17])
    data = featurizer(smiles)
    data = rw(data)

    return data


def _smiles_list_to_dataloader(smiles_list: list[str], batch_size: int = 2) -> DataLoader:
    data_list = [_smiles_to_pyg_data(smiles) for smiles in smiles_list]
    collate_fn = DefaultCollater(follow_batch=None, exclude_keys=None)
    dataloader = DataLoader(data_list, batch_size=batch_size, collate_fn=collate_fn)
    return dataloader


def load_model_components(
    ckpt_path: Path, encoder_config_path: Path, task_head_config_path: Path
) -> tuple[nn.Module, dict[str, nn.Module], dict[str, Callable | None]]:
    ckpt = torch.load(ckpt_path)
    encoder_config = OmegaConf.load(encoder_config_path)
    encoder = hydra.utils.instantiate(encoder_config)["module"]
    encoder_dict = {}
    for key in ckpt["state_dict"].keys():
        if "encoder" in key:
            encoder_dict[".".join(key.split(".")[1:])] = ckpt["state_dict"][key]
    encoder.load_state_dict(encoder_dict)
    encoder.eval()
    encoder.to(device)

    multitask_config_path = task_head_config_path
    task_head_path = multitask_config_path.parent.parent
    task_head_configs = OmegaConf.load(task_head_config_path)

    task_heads = {}
    task_head_post_processes = {}
    for task_head_config in task_head_configs["defaults"]:
        task_head_config_path = task_head_path / task_head_config
        task_head_config = OmegaConf.load(task_head_config_path)
        task_name = list(task_head_config.keys())[0]
        task_head_config[task_name]["module"]["in_features"] = encoder_config["module"]["d_model"]

        task_head_instance = hydra.utils.instantiate(task_head_config)[task_name]

        task_head = task_head_instance["module"]
        task_head_dict = {}
        for key in ckpt["state_dict"].keys():
            if task_name in key:
                task_head_dict[".".join(key.split(".")[2:])] = ckpt["state_dict"][key]
        _load_result = task_head.load_state_dict(task_head_dict)
        print(f"Loading task head {task_name} result: {_load_result}")
        task_head.eval()
        task_head.to(device)
        task_heads.update({task_name: task_head})

        task_head_post_process = task_head_instance.get("post_process")
        task_head_post_processes.update({task_name: task_head_post_process})
    return encoder, task_heads, task_head_post_processes


def main(
    smiles_list: list[str],
    ckpt_path: Path,
    encoder_config_path: Path,
    task_head_config_path: Path,
    device: torch.device = torch.device("cuda"),
) -> pd.DataFrame:
    encoder, task_heads, task_head_post_processes = load_model_components(
        ckpt_path, encoder_config_path, task_head_config_path
    )
    inference_module = InferenceModule(encoder, task_heads, task_head_post_processes, device)

    dataloader = _smiles_list_to_dataloader(smiles_list, batch_size=2)
    df_list = []
    for batch in dataloader:
        res = inference_module(batch)
        df = pd.DataFrame({key: value.cpu().flatten() for key, value in res.items()})
        df_list.append(df)

    df_qip = pd.concat(df_list)
    df_qip.insert(0, "smiles", smiles_list)
    df_qip.insert(
        1,
        "inchikey",
        [Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)) for smiles in smiles_list],
    )
    return df_qip


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv_path", type=Path, help="Path to csv file")
    parser.add_argument("--smiles_column", type=str, help="Column name of smiles", default="smiles")
    parser.add_argument("--output_path", type=Path, help="Path to save result", default="qip_result.csv")
    parser.add_argument(
        "--ckpt_path", type=Path, default=Path("saved_model/multitask_weight_HAD.ckpt"), help="Path to model checkpoint"
    )
    parser.add_argument(
        "--encoder_config",
        type=Path,
        default=Path("configs/system/encoder_config/gps/medium.yaml"),
        help="Path to encoder config",
    )
    parser.add_argument(
        "--task_head_config",
        type=Path,
        default=Path("configs/system/task_head_configs/gps/MT0.yaml"),
        help="Path to task head config",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        print("CUDA is available")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_qip = main(
        pd.read_csv(args.csv_path)[args.smiles_column].tolist(),
        args.ckpt_path,
        args.encoder_config,
        args.task_head_config,
        device=torch.device(device),
    )
    df_qip.to_csv(args.output_path, index=False)
