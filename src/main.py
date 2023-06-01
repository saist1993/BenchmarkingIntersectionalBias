# Setting up logging system. In this projects all results are parsed from the log file.
import torch
import logging
import shortuuid
import misc_utils
from pathlib import Path
from typing import Optional, NamedTuple

LOG_DIR = Path("../logs")
SAVED_MODEL_PATH = Path("../saved_models/")


class RunnerArguments(NamedTuple):
    """Arguments for the main function
    @TODO: clean this arguments files a bit!
    """
    seed: int
    dataset_name: str
    batch_size: int
    model: str
    epochs: int = 20
    save_model_as: Optional[str] = None
    method: str = 'unconstrained'
    optimizer_name: str = 'adam'
    lr: float = 0.001
    use_wandb: bool = False  # For legacy purpose. Not in the current codebase
    adversarial_lambda: float = 20.0
    dataset_size: int = 10000
    attribute_id: Optional[int] = None
    fairness_lambda: float = 0.0
    log_file_name: Optional[str] = None
    fairness_function: str = 'equal_opportunity'
    titled_t: float = 5.0
    mixup_rg: float = 0.5
    max_number_of_generated_examples: float = 1.0
    use_dropout: float = 0.2  # 0.0 corresponds to no dropout being applied
    use_batch_norm: float = 0.0  # 0.0 corresponds to no batch norm being applied
    per_group_label_number_of_examples: int = 1000
    positive_gen_model: str = "gen_model_positive_numeracy_10_simple.pt"
    negative_gen_model: str = "gen_model_negative_numeracy_10_simple.pt"
    log_dir: str = LOG_DIR


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


if __name__ == "__main__":
    pass
