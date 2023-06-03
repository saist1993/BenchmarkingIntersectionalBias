import torch
import logging
from tqdm.auto import tqdm
from .training_utils import get_iterators
from arguments import TrainingLoopParameters, ParsedDataset, SimpleTrainParameters


def train(train_parameters: SimpleTrainParameters):
    model, optimizer, device, criterion = \
        train_parameters.model, train_parameters.optimizer, train_parameters.device, train_parameters.criterion

    model.train()

    track_output = []

    for items in tqdm(train_parameters.iterator):
        for key in items.keys():
            items[key] = items[key].to(device)

        optimizer.zero_grad()
        output = model(items)
        loss = criterion(output['prediction'], items['labels'], items['aux_flattened'], mode='train')
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()
        track_output.append(output)

    model.eval()

    return None, None


def orchestrator(training_loop_parameters: TrainingLoopParameters, parsed_dataset: ParsedDataset):
    """
    :param training_loop_parameters: contains all information needed to train the model and the information related to settings
    :param parsed_dataset: contains all information related to dataset.
    :return:
    """
    if training_loop_parameters.log_run:
        logger = logging.getLogger(training_loop_parameters.unique_id_for_run)
    else:
        logger = False

    output = {}
    all_train_eps_metrics = []
    all_test_eps_metrics = []
    all_valid_eps_metrics = []

    # create the iterator
    original_train_iterator, train_iterator, valid_iterator, test_iterator = get_iterators(training_loop_parameters,
                                                                                           parsed_dataset)

    for ep in range(training_loop_parameters.n_epochs):
        if logger: logger.info("start of epoch block")

        train_params = SimpleTrainParameters(
            model=training_loop_parameters.model,
            iterator=train_iterator,
            optimizer=training_loop_parameters.optimizer,
            criterion=training_loop_parameters.criterion,
            device=training_loop_parameters.device,
            other_params={},
            per_epoch_metric=None,
            fairness_function=training_loop_parameters.fairness_function)

        _, _ = train(train_parameters=train_params)






