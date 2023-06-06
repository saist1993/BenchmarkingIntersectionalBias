import torch
import logging
import numpy as np
from metrics import epoch_metric
from sklearn.linear_model import LogisticRegression
from .simple_training_loop import orchestrator as simple_orch
from .inlp_training_loop_utils import get_debiasing_projection, get_projection_to_intersection_of_nullspaces
from arguments import TrainingLoopParameters, ParsedDataset, SimpleTrainParameters, EpochMetric
from .training_utils import get_iterators, collect_output, get_fairness_related_meta_dict, mixup_sub_routine


def get_hidden_representation_aux(iterator, model, device):
    model.eval()
    hidden_representations = []
    aux = []
    labels = []
    for items in iterator:
        for key in items.keys():
            items[key] = items[key].to(device)
        output = model(items)
        hidden_representations.append(output['hidden'].detach().numpy())
        aux.append(items['aux'].numpy())
        labels.append(items['labels'].numpy())

    aux = np.vstack(aux)
    label = np.hstack(labels)
    hidden_representations = np.vstack(hidden_representations)

    return hidden_representations, aux, label


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

    original_train_iterator, train_iterator, valid_iterator, test_iterator = get_iterators(training_loop_parameters,
                                                                                           parsed_dataset,
                                                                                           shuffle=True)

    # Step1 - Train a regular model for few epochs
    assert training_loop_parameters.method == "inlp"
    training_loop_parameters.log_run = False
    training_loop_parameters.method = "unconstrained"
    training_loop_parameters.n_epochs = 5
    training_loop_parameters.model = simple_orch(training_loop_parameters, parsed_dataset)
    training_loop_parameters.method = "inlp"

    # Step2: Extract Hidden representations
    train_hidden, train_s, train_labels = get_hidden_representation_aux(train_iterator, training_loop_parameters.model,
                                                                        training_loop_parameters.device)
    valid_hidden, valid_s, valid_labels = get_hidden_representation_aux(valid_iterator, training_loop_parameters.model,
                                                                        training_loop_parameters.device)
    test_hidden, test_s, test_labels = get_hidden_representation_aux(valid_iterator, training_loop_parameters.model,
                                                                     training_loop_parameters.device)

    # Step3: Setup a classifier
    discriminator_reweighting = None
    n = 15
    min_acc = 0.0
    is_autoregressive = True
    dim = train_hidden.shape[1]  # @TODO: This needs to be set `
    clf = LogisticRegression
    clf_params = {'fit_intercept': True, 'class_weight': discriminator_reweighting, 'dual': False, 'C': 0.1,
                  "max_iter": 500}

    # Step4: Run INLP
    P_n = get_debiasing_projection(clf, clf_params, n, dim, is_autoregressive, min_acc,
                                   train_hidden, train_s, valid_hidden, valid_s)

    rowspaces = P_n[1]

    for iteration, p_iteration in enumerate(range(1, len(rowspaces))):
        if logger: logger.info("start of epoch block")

        P = get_projection_to_intersection_of_nullspaces(rowspaces[:p_iteration], input_dim=train_hidden.shape[1])

        debiased_x_train = P.dot(train_hidden.T).T
        debiased_x_dev = P.dot(valid_hidden.T).T
        debiased_x_test = P.dot(test_hidden.T).T

        classifier = LogisticRegression(warm_start=True,
                                        penalty='l2',
                                        solver="sag",
                                        multi_class='multinomial',
                                        fit_intercept=True,
                                        verbose=0,
                                        max_iter=200,
                                        n_jobs=24,
                                        random_state=1)
        classifier.fit(debiased_x_train, train_labels)
        train_y_pred = classifier.predict(debiased_x_train)
        valid_y_pred = classifier.predict(debiased_x_dev)
        test_y_pred = classifier.predict(debiased_x_test)

        train_em = EpochMetric(
            predictions=train_y_pred,
            labels=train_labels,
            s=train_s,
            fairness_function=training_loop_parameters.fairness_function)
        train_emo = epoch_metric.CalculateEpochMetric(epoch_metric=train_em).run()
        train_emo.epoch_number = iteration
        if logger: logger.info(f"train epoch metric: {train_emo}")

        valid_em = EpochMetric(
            predictions=valid_y_pred,
            labels=valid_labels,
            s=valid_s,
            fairness_function=training_loop_parameters.fairness_function)
        valid_emo = epoch_metric.CalculateEpochMetric(epoch_metric=valid_em).run()
        valid_emo.epoch_number = iteration
        if logger: logger.info(f"valid epoch metric: {valid_emo}")

        test_em = EpochMetric(
            predictions=test_y_pred,
            labels=test_labels,
            s=test_s,
            fairness_function=training_loop_parameters.fairness_function)
        test_emo = epoch_metric.CalculateEpochMetric(epoch_metric=test_em).run()
        test_emo.epoch_number = iteration
        if logger: logger.info(f"test epoch metric: {test_emo}")

        if logger: logger.info("end of epoch block")
