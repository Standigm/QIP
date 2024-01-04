import os
import sys
import socket

import dotenv
import hydra
from omegaconf import DictConfig
from admet_prediction.utils.misc import get_logger

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


log = get_logger(__name__)


@hydra.main(version_base="1.3", config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    import admet_prediction
    from admet_prediction.deployment import deployment
    from admet_prediction.inference import inference
    from admet_prediction.test import test
    from admet_prediction.train import train
    from admet_prediction.utils.misc import print_config

    # print hostname
    try:
        log.info(f"Host name: {socket.gethostname()}")
        log.info(f"Host IP address: {socket.gethostbyname(socket.gethostname())}")
    except:
        pass
    # print slurm job_id if exists
    if os.environ.get("SLURM_JOB_ID", None) is not None:
        log.info(f'SLURM_JOB_ID: {os.environ.get("SLURM_JOB_ID")}')

    # pring configs
    print_config(config, resolve=True)

    # TODO: Add model export, test, inference(feature_extraction),
    if config.mode == "train":
        # Train model
        return train(config)
    elif config.mode == "test":
        return test(config)
    elif config.mode == "inference":
        return inference(config)
    elif config.mode == "deployment":
        return deployment(config)
    else:
        raise ValueError(f"Invalid config.mode == {config.mode}")


if __name__ == "__main__":
    main()
