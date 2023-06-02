# Setting up logging system. In this projects all results are parsed from the log file.
import torch
import logging
import shortuuid
import misc_utils
from pathlib import Path
from arguments import RunnerArguments
from typing import Optional, NamedTuple

LOG_DIR = Path("../logs")
SAVED_MODEL_PATH = Path("../saved_models/")


def runner(runner_arguments: RunnerArguments):
    """Does the whole heavy lifting for a given setting a.k.a method+seed+dataset+other_parameters"""

    misc_utils.set_seed(runner_arguments.seed)
    # device = resolve_device()
    device = torch.device('cpu')

    # setup unique id for the run
    unique_id_for_run = shortuuid.uuid()

    misc_utils.get_logger(unique_id_for_run, runner_arguments.log_file_name, runner_arguments.log_dir, runner_arguments)
    logger = logging.getLogger(str(unique_id_for_run))
    logger.info(f"unique id is:{unique_id_for_run}")

    logger.info(f"arguments: {locals()}")

    iterator_params = {
        'batch_size': runner_arguments.batch_size,
        'lm_encoder_type': 'bert-base-uncased',
        'lm_encoding_to_use': 'use_cls',
        'seed': runner_arguments.seed,
        'standard_scalar': runner_arguments.standard_scalar
    }
    parsed_dataset = misc_utils.generate_raw_dataset(dataset_name="twitter_hate_speech", **iterator_params)


if __name__ == "__main__":
    runner_arguments = RunnerArguments()
    runner(runner_arguments)
