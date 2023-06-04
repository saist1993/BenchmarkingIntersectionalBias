import torch
import logging
import numpy as np
from tqdm.auto import tqdm
from metrics import epoch_metric
from fairgrad.torch import CrossEntropyLoss as fairgrad_CrossEntropyLoss
from arguments import TrainingLoopParameters, ParsedDataset, SimpleTrainParameters, EpochMetric
from .training_utils import get_iterators, collect_output, get_fairness_related_meta_dict, mixup_sub_routine


def train_simple(train_parameters: SimpleTrainParameters):
    model, optimizer, device, criterion = \
        train_parameters.model, train_parameters.optimizer, train_parameters.device, train_parameters.criterion

    model.train()

    track_output = []
    track_input = []

    for items in tqdm(train_parameters.iterator):
        for key in items.keys():
            items[key] = items[key].to(device)

        optimizer.zero_grad()
        output = model(items)
        loss = criterion(output['prediction'], items['labels'], items['aux_flattened'], mode='train')
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()
        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(items)

    model.eval()
    predictions, labels, s, loss = collect_output(all_batch_outputs=track_output, all_batch_inputs=track_input)
    em = EpochMetric(
        predictions=predictions,
        labels=labels,
        s=s,
        fairness_function=train_parameters.fairness_function)

    emo = epoch_metric.CalculateEpochMetric(epoch_metric=em).run()
    emo.loss = loss

    return emo


def train_group(train_parameters: SimpleTrainParameters):
    model, optimizer, device, criterion = \
        train_parameters.model, train_parameters.optimizer, train_parameters.device, train_parameters.criterion

    model.train()

    track_output = []
    track_input = []

    for _ in tqdm(range(train_parameters.number_of_iterations)):
        items = train_parameters.iterator()
        for key in items.keys():
            items[key] = items[key].to(device)

        optimizer.zero_grad()
        output = model(items)
        loss = criterion(output['prediction'], items['labels'], items['aux_flattened'], mode='train')
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()
        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(items)

    model.eval()
    predictions, labels, s, loss = collect_output(all_batch_outputs=track_output, all_batch_inputs=track_input)
    em = EpochMetric(
        predictions=predictions,
        labels=labels,
        s=s,
        fairness_function=train_parameters.fairness_function)

    emo = epoch_metric.CalculateEpochMetric(epoch_metric=em).run()
    emo.loss = loss

    return emo


def train_mixup_regularizer(train_parameters: SimpleTrainParameters):
    model, optimizer, device, criterion = \
        train_parameters.model, train_parameters.optimizer, train_parameters.device, train_parameters.criterion

    model.train()

    track_output = []
    track_input = []

    for _ in tqdm(range(train_parameters.number_of_iterations)):
        index = np.random.choice(len(train_parameters.all_unique_groups), 2, False)
        group1, group2 = train_parameters.all_unique_groups[index[0]], train_parameters.all_unique_groups[index[1]]
        items_group_1 = train_parameters.iterator(group1)
        items_group_2 = train_parameters.iterator(group2)

        for key in items_group_1.keys():
            items_group_1[key] = items_group_1[key].to(device)

        for key in items_group_2.keys():
            items_group_2[key] = items_group_2[key].to(device)

        optimizer.zero_grad()
        output_group_1 = model(items_group_1)
        output_group_2 = model(items_group_2)
        loss_group_1 = criterion(output_group_1['prediction'], items_group_1['labels'], items_group_1['aux_flattened'],
                                 mode='train')
        loss_group_2 = criterion(output_group_2['prediction'], items_group_2['labels'], items_group_2['aux_flattened'],
                                 mode='train')

        reg_loss = train_parameters.regularization_lambda * mixup_sub_routine(train_parameters, items_group_1,
                                                                              items_group_2, model)
        loss = torch.mean(loss_group_1) + torch.mean(loss_group_2) + reg_loss

        loss.backward()
        optimizer.step()

        output_group_1['loss_batch'] = loss.item()
        track_output.append(output_group_1)
        track_input.append(items_group_1)

    model.eval()
    predictions, labels, s, loss = collect_output(all_batch_outputs=track_output, all_batch_inputs=track_input)
    em = EpochMetric(
        predictions=predictions,
        labels=labels,
        s=s,
        fairness_function=train_parameters.fairness_function)

    emo = epoch_metric.CalculateEpochMetric(epoch_metric=em).run()
    emo.loss = loss

    return emo


def train_adversarial_single(train_parameters: SimpleTrainParameters):
    model, optimizer, device, criterion = \
        train_parameters.model, train_parameters.optimizer, train_parameters.device, train_parameters.criterion

    model.train()

    track_output = []
    track_input = []

    for items in tqdm(train_parameters.iterator):
        for key in items.keys():
            items[key] = items[key].to(device)

        optimizer.zero_grad()
        output = model(items)
        loss = criterion(output['prediction'], items['labels'], items['aux_flattened'], mode='train')
        adv_loss = train_parameters.regularization_lambda * torch.mean(
            criterion(output['adv_outputs'][0], items['aux_flattened']))
        loss = torch.mean(loss) + adv_loss
        loss.backward()
        optimizer.step()
        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(items)

    model.eval()
    predictions, labels, s, loss = collect_output(all_batch_outputs=track_output, all_batch_inputs=track_input)
    em = EpochMetric(
        predictions=predictions,
        labels=labels,
        s=s,
        fairness_function=train_parameters.fairness_function)

    emo = epoch_metric.CalculateEpochMetric(epoch_metric=em).run()
    emo.loss = loss

    return emo


def test(train_parameters: SimpleTrainParameters):
    model, optimizer, device, criterion = \
        train_parameters.model, train_parameters.optimizer, train_parameters.device, train_parameters.criterion

    model.eval()

    track_output = []
    track_input = []

    for items in tqdm(train_parameters.iterator):
        for key in items.keys():
            items[key] = items[key].to(device)

        optimizer.zero_grad()
        output = model(items)
        loss = criterion(output['prediction'], items['labels'], items['aux_flattened'], mode='train')
        loss = torch.mean(loss)
        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(items)

    predictions, labels, s, loss = collect_output(all_batch_outputs=track_output, all_batch_inputs=track_input)
    em = EpochMetric(
        predictions=predictions,
        labels=labels,
        s=s,
        fairness_function=train_parameters.fairness_function)

    emo = epoch_metric.CalculateEpochMetric(epoch_metric=em).run()
    emo.loss = loss

    return emo


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
    if training_loop_parameters.method == "mixup_regularizer":
        assert training_loop_parameters.iterator_type == "group_iterator"
        shuffle = False
    else:
        shuffle = True

    original_train_iterator, train_iterator, valid_iterator, test_iterator = get_iterators(training_loop_parameters,
                                                                                           parsed_dataset,
                                                                                           shuffle=shuffle)

    if training_loop_parameters.method == "fairgrad":
        assert training_loop_parameters.iterator_type == "simple_iterator"
        fairness_related_meta_data = get_fairness_related_meta_dict(train_iterator,
                                                                    training_loop_parameters.fairness_function,
                                                                    fairness_rate=0.01,
                                                                    epsilon=0.0)

        training_loop_parameters.criterion = fairgrad_CrossEntropyLoss(reduction='none', **fairness_related_meta_data)

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
            fairness_function=training_loop_parameters.fairness_function,
            number_of_iterations=training_loop_parameters.number_of_iterations,
            all_unique_groups=parsed_dataset.all_groups,
            regularization_lambda=training_loop_parameters.regularization_lambda,
            batch_size=training_loop_parameters.batch_size)

        if training_loop_parameters.method == "mixup_regularizer":
            train_emo = train_mixup_regularizer(train_parameters=train_params)
        elif training_loop_parameters.method == "adversarial_single":
            train_emo = train_adversarial_single(train_parameters=train_params)
        else:
            if training_loop_parameters.iterator_type == "simple_iterator":
                train_emo = train_simple(train_parameters=train_params)
            elif training_loop_parameters.iterator_type == "group_iterator":
                train_emo = train_group(train_parameters=train_params)
            else:
                raise NotImplementedError

        train_emo.epoch_number = ep
        if logger: logger.info(f"train epoch metric: {train_emo}")

        valid_params = SimpleTrainParameters(
            model=training_loop_parameters.model,
            iterator=valid_iterator,
            optimizer=training_loop_parameters.optimizer,
            criterion=training_loop_parameters.criterion,
            device=training_loop_parameters.device,
            other_params={},
            per_epoch_metric=None,
            fairness_function=training_loop_parameters.fairness_function)

        valid_emo = test(train_parameters=valid_params)
        valid_emo.epoch_number = ep
        if logger: logger.info(f"valid epoch metric: {valid_emo}")

        test_params = SimpleTrainParameters(
            model=training_loop_parameters.model,
            iterator=test_iterator,
            optimizer=training_loop_parameters.optimizer,
            criterion=training_loop_parameters.criterion,
            device=training_loop_parameters.device,
            other_params={},
            per_epoch_metric=None,
            fairness_function=training_loop_parameters.fairness_function)

        test_emo = test(train_parameters=test_params)
        test_emo.epoch_number = ep
        if logger: logger.info(f"test epoch metric: {test_emo}")

        if logger: logger.info("end of epoch block")

    return training_loop_parameters.model
