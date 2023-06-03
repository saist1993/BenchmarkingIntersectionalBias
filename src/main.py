# Setting up logging system. In this projects all results are parsed from the log file.
import torch
import logging
import arguments
import shortuuid
import misc_utils
from pathlib import Path
from typing import Optional
from training_loops import simple_training_loop
from fairgrad.torch import CrossEntropyLoss as fairgrad_CrossEntropyLoss

LOG_DIR = Path("../logs")
SAVED_MODEL_PATH = Path("../saved_models/")


def runner(runner_arguments: arguments.RunnerArguments):
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

    # get parsed dataset
    iterator_params = {
        'batch_size': runner_arguments.batch_size,
        'lm_encoder_type': 'bert-base-uncased',
        'lm_encoding_to_use': 'use_cls',
        'seed': runner_arguments.seed,
        'standard_scalar': runner_arguments.standard_scalar
    }
    parsed_dataset = misc_utils.generate_raw_dataset(dataset_name="twitter_hate_speech", **iterator_params)

    # get model
    model = misc_utils.get_model(
        method=runner_arguments.method,
        model_name=runner_arguments.model,
        other_meta_data=parsed_dataset,
        attribute_id=runner_arguments.attribute_id,
        device=device,
        use_batch_norm=runner_arguments.use_batch_norm,
        use_dropout=runner_arguments.use_dropout
    )

    # get optimizer
    if runner_arguments.optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=runner_arguments.lr)
    elif runner_arguments.optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=runner_arguments.lr)

    # get criterion
    criterion = fairgrad_CrossEntropyLoss(reduction='none')

    training_loop_params = arguments.TrainingLoopParameters(
        n_epochs=runner_arguments.epochs,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        other_params={},
        save_model_as=None,
        fairness_function=runner_arguments.fairness_function,
        unique_id_for_run=unique_id_for_run,
        batch_size=runner_arguments.batch_size,
        per_group_size=runner_arguments.per_group_label_number_of_examples,
        log_run=True,
        iterator_type=runner_arguments.iterator_type,
        number_of_iterations=int(parsed_dataset.train_X.shape[0] / runner_arguments.batch_size),
        method=runner_arguments.method,
        regularization_lambda=runner_arguments.regularization_lambda)

    simple_training_loop.orchestrator(training_loop_parameters=training_loop_params, parsed_dataset=parsed_dataset)


if __name__ == "__main__":
    runner_arguments = arguments.RunnerArguments()
    runner(runner_arguments)
