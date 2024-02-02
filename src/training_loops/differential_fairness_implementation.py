import torch
import logging
import numpy as np
from tqdm.auto import tqdm
from metrics import epoch_metric
# from custom_fairness_loss import FairnessLoss
from fairgrad.torch import CrossEntropyLoss as fairgrad_CrossEntropyLoss
from arguments import TrainingLoopParameters, ParsedDataset, SimpleTrainParameters, \
    EpochMetric
from .training_utils import get_iterators, collect_output, \
    get_fairness_related_meta_dict, mixup_sub_routine

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from .differnetial_fairness_utils import computeEDFforData
from .differnetial_fairness_utils import NeuralNet, training_fair_model


def if_alpha(min_eps, max_eps, beta, fairness_function="equal_odds"):
    """
        At alpha=0 or beta=0 we only care about delta_rel
        At alpha=1 or beta=\infity we only care about delta_abs
    """

    if fairness_function == "equal_odds":
        min_eps, max_eps = 1 - max_eps, 1 - min_eps  # becuase of FPR wanting to maximize

    assert max_eps > min_eps

    delta_rel = (1.0 - max_eps) / (1.0 - min_eps)  # BETWEEN 0 AND 1

    delta_abs = max(1 - min_eps, 1 - max_eps)

    return (1 - beta) * delta_rel + beta * delta_abs


def df_eps(min_eps, max_eps, fairness_function="equal_odds"):
    if fairness_function == "equal_odds":
        min_eps, max_eps = 1 - max_eps, 1 - min_eps  # becuase of FPR wanting to maximize
    return np.log(max(min_eps, max_eps) / min(min_eps, max_eps))


def parse_emo(emo: epoch_metric.CalculateEpochMetric, fairness_function):
    print(f"accuracy is {emo.accuracy}")
    print(f"balanced accuracy is {emo.balanced_accuracy}")
    intersectional_bootstrap = emo.eps_fairness[
        fairness_function].intersectional_bootstrap

    computed_metric = intersectional_bootstrap[10]
    eps = [i[-1] for i in computed_metric]
    print(eps)
    if_fairness = if_alpha(min(eps), max(eps), 0.5, fairness_function=fairness_function)
    df_fairness = df_eps(min(eps), max(eps), fairness_function="equal_odds")
    print(f"IF-0.5 is {if_fairness}")
    print(f"DF is {df_fairness}")
    if fairness_function == "equal_odds":
        print(f"{1 - min(eps), 1 - max(eps)}")
        max_eps = 1 - min(eps)
        min_eps = 1 - max(eps)
    elif fairness_function == "equal_opportunity":
        print(f"{max(eps), min(eps)}")
        max_eps = max(eps)
        min_eps = min(eps)
    else:
        raise NotImplementedError

    return emo.balanced_accuracy, max_eps, min_eps, df_fairness, if_fairness


def orchestrator(training_loop_parameters: TrainingLoopParameters,
                 parsed_dataset: ParsedDataset):
    """
    :param training_loop_parameters: contains all information needed to train the model and the information related to settings
    :param parsed_dataset: contains all information related to dataset.
    :return:
    """
    if training_loop_parameters.log_run:
        logger = logging.getLogger(training_loop_parameters.unique_id_for_run)
    else:
        logger = False

    pd = parsed_dataset

    X, dev_X, y, dev_y, S, dev_S = pd.train_X, pd.valid_X, pd.train_y, pd.valid_y, pd.train_s, pd.valid_s
    test_X = pd.test_X
    test_y = pd.test_y
    test_S = pd.test_s
    print("integrating custom DF implementation!")

    intersectionalGroups = np.unique(S, axis=0)

    # deep neural network using pytorch
    trainData = torch.from_numpy(X)
    trainLabel = torch.from_numpy(y.reshape((-1, 1)))

    # devData = torch.from_numpy(devData)
    testData = torch.from_numpy(test_X)
    devData = torch.from_numpy(dev_X)

    # hyperparameters
    input_size = trainData.size()[1]
    hidden1 = 16
    hidden2 = 16
    hidden3 = 16
    output_size = 1
    miniBatch = training_loop_parameters.batch_size  # mini-batch size
    num_epochs = training_loop_parameters.n_epochs
    stepSize = 0.1
    learning_rate = 0.001
    burnIn = 5
    epsilonBase = torch.tensor(
        0.0)  # To protect the 80%-rule in intersectional setting, set this variable to: - log(0.8) = 0.2231

    # %%

    # %% training DNN model with fairness constraint

    # Train a fair classifier
    lamda = torch.tensor(
        0.01)  # λ is ahyper-parameter that balances between the prediction loss and fairness.
    # Select λ for fair learning algorithms via rigorous grid search on the development set. See paper for details.
    DF_Model = training_fair_model(training_loop_parameters.model,
                                   training_loop_parameters.criterion,
                                   learning_rate, num_epochs, trainData,
                                   trainLabel, miniBatch, S, intersectionalGroups,
                                   burnIn, stepSize, epsilonBase, lamda)

    # %%
    # Validate the model
    with torch.no_grad():
        devData = Variable(devData.float())
        predictProb = DF_Model(devData)
        # predicted = ((predictProb > 0.5).numpy()).reshape((-1,))
        predicted = torch.argmax(predictProb, 1)
        predicted = predicted.detach().numpy()
        Accuracy = sum(predicted == dev_y) / len(dev_y)

    # Save results

    # predictProb = (predictProb.numpy()).reshape((-1,))
    #
    # print(f"DF classifier dev accuracy: {Accuracy: .3f}")
    # aucScore = roc_auc_score(dev_y, predictProb)
    # print(f"DF classifier dev ROC AUC: {aucScore: .3f}")
    # nn_f1 = f1_score(dev_y, predicted)
    # print(f"DF classifier dev F1 score: {nn_f1: .2f}")
    #
    # epsilon_hard, epsilon_soft, gamma_hard, gamma_soft, p_rule_hard, p_rule_soft = computeEDFforData(
    #     dev_S, predicted, predictProb, intersectionalGroups)
    #
    # print(f"DF classifier dev epsilon_hard: {epsilon_hard: .3f}")
    # print(f"DF classifier dev epsilon_soft: {epsilon_soft: .3f}")
    # print(f"DF classifier dev gamma_hard: {gamma_hard: .3f}")
    # print(f"DF classifier dev gamma_soft: {gamma_soft: .3f}")
    # print(f"DF classifier dev p_rule_hard: {p_rule_hard: .3f}")
    # print(f"DF classifier dev p_rule_soft: {p_rule_soft: .3f}")
    # %%
    # Test the model
    with torch.no_grad():
        testData = Variable(testData.float())
        predictProb = DF_Model(testData)
        predicted = torch.argmax(predictProb, 1)
        predicted = predicted.detach().numpy()
        Accuracy = sum(predicted == test_y) / len(test_y)

    em = EpochMetric(
        predictions=predicted,
        labels=test_y,
        s=test_S,
        fairness_function=training_loop_parameters.fairness_function)

    emo = epoch_metric.CalculateEpochMetric(epoch_metric=em).run()

    return parse_emo(emo, training_loop_parameters.fairness_function)

    # Save results

    # predictProb = (predictProb.numpy()).reshape((-1,))
    #
    # print(f"DF_Classifier accuracy: {Accuracy: .3f}")
    # aucScore = roc_auc_score(test_y, predictProb)
    # print(f"DF_Classifier ROC AUC: {aucScore: .3f}")
    # nn_f1 = f1_score(test_y, predicted)
    # print(f"DF_Classifier F1 score: {nn_f1: .2f}")
    #
    # epsilon_hard, epsilon_soft, gamma_hard, gamma_soft, p_rule_hard, p_rule_soft = computeEDFforData(
    #     test_S, predicted, predictProb, intersectionalGroups)
    #
    # print(f"DF_Classifier epsilon_hard: {epsilon_hard: .3f}")
    # print(f"DF_Classifier epsilon_soft: {epsilon_soft: .3f}")
    # print(f"DF_Classifier gamma_hard: {gamma_hard: .3f}")
    # print(f"DF_Classifier gamma_soft: {gamma_soft: .3f}")
    # print(f"DF_Classifier p_rule_hard: {p_rule_hard: .3f}")
    # print(f"DF_Classifier p_rule_soft: {p_rule_soft: .3f}")
