# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:03:22 2020

@author: islam
"""
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


# Loss and optimizer
def fairness_loss(base_fairness, stochasticModel):
    # DF-based penalty term
    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter / numClasses
    zeroTerm = torch.tensor(0.0)

    theta = (stochasticModel.countClass_hat + dirichletAlpha) / (
            stochasticModel.countTotal_hat + concentrationParameter)
    epsilonClass = differentialFairnessBinaryOutcomeTrain(theta)
    return torch.max(zeroTerm, (epsilonClass - base_fairness))


# Loss and optimizer
def sf_loss(base_fairness, stochasticModel):
    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter / numClasses
    zeroTerm = torch.tensor(0.0)
    population = sum(stochasticModel.countTotal_hat).detach()

    theta = (stochasticModel.countClass_hat + dirichletAlpha) / (
            stochasticModel.countTotal_hat + concentrationParameter)
    alpha = (stochasticModel.countTotal_hat + dirichletAlpha) / (
            population + concentrationParameter)
    gammaClass = subgroupFairnessTrain(theta, alpha)
    return torch.max(zeroTerm, (gammaClass - base_fairness))


def prule_loss(base_fairness, stochasticModel):
    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter / numClasses
    zeroTerm = torch.tensor(0.0)

    theta_minority = (stochasticModel.countClass_hat[0] + dirichletAlpha) / (
            stochasticModel.countTotal_hat[0] + concentrationParameter)
    theta_majority = (stochasticModel.countClass_hat[1] + dirichletAlpha) / (
            stochasticModel.countTotal_hat[1] + concentrationParameter)
    pruleClass = torch.min(theta_minority / theta_majority,
                           theta_majority / theta_minority) * 100.0
    return torch.max(zeroTerm, (base_fairness - pruleClass))


# %%
# Measure intersectional DF from positive predict probabilities
def differentialFairnessBinaryOutcomeTrain(probabilitiesOfPositive):
    # input: probabilitiesOfPositive = positive p(y|S) from ML algorithm
    # output: epsilon = differential fairness measure
    epsilonPerGroup = torch.zeros(len(probabilitiesOfPositive), dtype=torch.float)
    for i in range(len(probabilitiesOfPositive)):
        epsilon = torch.tensor(0.0)  # initialization of DF
        for j in range(len(probabilitiesOfPositive)):
            if i == j:
                continue
            else:
                epsilon = torch.max(epsilon, torch.abs(
                    torch.log(probabilitiesOfPositive[i]) - torch.log(
                        probabilitiesOfPositive[
                            j])))  # ratio of probabilities of positive outcome
                epsilon = torch.max(epsilon, torch.abs(
                    (torch.log(1 - probabilitiesOfPositive[i])) - (torch.log(
                        1 - probabilitiesOfPositive[
                            j]))))  # ratio of probabilities of negative outcome
        epsilonPerGroup[i] = epsilon  # DF per group
    epsilon = torch.max(epsilonPerGroup)  # overall DF of the algorithm
    return epsilon


def subgroupFairnessTrain(probabilitiesOfPositive, alphaSP):
    # input: probabilitiesOfPositive = Pr[D(X)=1|g(x)=1]
    #        alphaG = Pr[g(x)=1]
    # output: gamma-unfairness
    spD = sum(
        probabilitiesOfPositive * alphaSP)  # probabilities of positive class across whole population SP(D) = Pr[D(X)=1]
    gammaPerGroup = torch.zeros(len(probabilitiesOfPositive),
                                dtype=torch.float)  # SF per group
    for i in range(len(probabilitiesOfPositive)):
        gammaPerGroup[i] = alphaSP[i] * torch.abs(spD - probabilitiesOfPositive[i])
    gamma = torch.max(gammaPerGroup)  # overall SF of the algorithm
    return gamma


# %% stochastic count updates
def computeBatchCounts(protectedAttributes, intersectGroups, predictions):
    # intersectGroups should be pre-defined so that stochastic update of p(y|S)
    # can be maintained correctly among different batches

    # compute counts for each intersectional group
    predictions = torch.argmax(predictions, 1)
    countsClassOne = torch.zeros((len(intersectGroups)), dtype=torch.float)
    countsTotal = torch.zeros((len(intersectGroups)), dtype=torch.float)
    for i in range(len(predictions)):
        index = np.where(
            (intersectGroups == protectedAttributes[i].detach().numpy()).all(axis=1))[
            0][0]
        countsTotal[index] = countsTotal[index] + 1
        countsClassOne[index] = countsClassOne[index] + predictions[i]
    return countsClassOne, countsTotal


def computeBatchCounts_false_positive_rate(protectedAttributes, intersectGroups,
                                           predictions, labels):
    predictions = torch.argmax(predictions, 1)
    countsClassOne = torch.zeros((len(intersectGroups)), dtype=torch.float)
    countsTotal = torch.zeros((len(intersectGroups)), dtype=torch.float)
    custom_counts = torch.zeros((len(intersectGroups)), dtype=torch.float)
    for i in range(len(predictions)):
        index = np.where(
            (intersectGroups == protectedAttributes[i].detach().numpy()).all(axis=1))[
            0][0]
        countsTotal[index] = countsTotal[index] + 1
        if predictions[i] == 1 and labels[i] == 0:
            countsClassOne[index] = countsClassOne[index] + 1

        if labels[i] == 0:
            custom_counts[index] = custom_counts[index] + 1

    countsClassOne = countsClassOne / custom_counts
    return countsClassOne, countsTotal


class stochasticCountModel(nn.Module):
    def __init__(self, no_of_groups, N, batch_size):
        super(stochasticCountModel, self).__init__()
        self.countClass_hat = torch.ones((no_of_groups))
        self.countTotal_hat = torch.ones((no_of_groups))

        self.countClass_hat = self.countClass_hat * (N / (batch_size * no_of_groups))
        self.countTotal_hat = self.countTotal_hat * (N / batch_size)

    def forward(self, rho, countClass_batch, countTotal_batch, N, batch_size):
        self.countClass_hat = (1 - rho) * self.countClass_hat + rho * (
                N / batch_size) * countClass_batch
        self.countTotal_hat = (1 - rho) * self.countTotal_hat + rho * (
                N / batch_size) * countTotal_batch


# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:00:04 2020

@author: islam
"""
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


# from DF_Training import stochasticCountModel, computeBatchCounts, fairness_loss, \
#     sf_loss, prule_loss

# from fairness_metrics import computeEDFforData


# fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, hidden3, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
        self.outputLayer = nn.Linear(hidden3, output_size)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.drop3(out)
        out = self.outputLayer(out)
        out = self.out_act(out)
        return out


# %% train the model without fairness constraint
def training_typical(input_size, hidden1, hidden2, hidden3, output_size, learning_rate,
                     num_epochs, trainData, trainLabel, miniBatch):
    dnn_model = NeuralNet(input_size, hidden1, hidden2, hidden3, output_size)
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(dnn_model.parameters(), lr=learning_rate)
    # optimizer = optim.Adadelta(modelFair.parameters())
    # optimizer = optim.Adamax(modelFair.parameters(),lr = learning_rate)
    # Train the netwok
    for epoch in range(num_epochs):
        for batch in range(0,
                           np.int64(np.floor(len(trainData) / miniBatch)) * miniBatch,
                           miniBatch):
            trainY_batch = trainLabel[batch:(batch + miniBatch)]
            trainX_batch = trainData[batch:(batch + miniBatch)]

            trainX_batch = Variable(trainX_batch.float())
            trainY_batch = Variable(trainY_batch.float())

            # forward + backward + optimize
            outputs = dnn_model(trainX_batch)
            tot_loss = criterion(outputs, trainY_batch)

            # zero the parameter gradients
            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()
        print('epoch: ', epoch, 'out of: ', num_epochs, 'average loss: ',
              tot_loss.item())
    return dnn_model


# %% train the model with differential fairness constraint
def training_fair_model(model, criterion,
                        learning_rate, num_epochs, trainData, trainLabel, miniBatch, S,
                        intersectionalGroups, burnIn, stepSize, epsilonBase, lamda):
    VB_CountModel = stochasticCountModel(len(intersectionalGroups), len(trainData),
                                         miniBatch)

    modelFair = model
    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelFair.parameters(), lr=learning_rate)
    # optimizer = optim.Adadelta(modelFair.parameters())
    # optimizer = optim.Adamax(modelFair.parameters(),lr = learning_rate)
    # Train the netwok
    for epoch in range(burnIn):
        for batch in range(0,
                           np.int64(np.floor(len(trainData) / miniBatch)) * miniBatch,
                           miniBatch):
            trainS_batch = S[batch:(
                    batch + miniBatch)]  # protected attributes in the mini-batch
            trainY_batch = trainLabel[batch:(batch + miniBatch)]
            trainX_batch = trainData[batch:(batch + miniBatch)]

            trainX_batch = Variable(trainX_batch.float())
            trainY_batch = Variable(trainY_batch.float())

            # forward + backward + optimize
            outputs = modelFair(trainX_batch)
            tot_loss = criterion(outputs, trainY_batch.long().squeeze())
            tot_loss = torch.mean(tot_loss)
            # zero the parameter gradients
            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()
        print('burn-in epoch: ', epoch, 'out of: ', burnIn, 'average loss: ',
              tot_loss.item())

    for epoch in range(num_epochs):

        for batch in range(0,
                           np.int64(np.floor(len(trainData) / miniBatch)) * miniBatch,
                           miniBatch):
            trainS_batch = S[batch:(
                    batch + miniBatch)]  # protected attributes in the mini-batch
            trainY_batch = trainLabel[batch:(batch + miniBatch)]
            trainX_batch = trainData[batch:(batch + miniBatch)]

            trainX_batch = Variable(trainX_batch.float())
            trainY_batch = Variable(trainY_batch.float())

            VB_CountModel.countClass_hat.detach_()
            VB_CountModel.countTotal_hat.detach_()
            # forward + backward + optimize
            outputs = modelFair(trainX_batch)
            loss = criterion(outputs, trainY_batch.long().squeeze())
            loss = torch.mean(loss)

            # update Count model
            countClass, countTotal = computeBatchCounts(trainS_batch,
                                                        intersectionalGroups,
                                                        torch.nn.functional.sigmoid(
                                                            outputs))
            # thetaModel(stepSize,theta_batch)
            VB_CountModel(stepSize, countClass, countTotal, len(trainData), miniBatch)

            # fairness constraint
            lossDF = fairness_loss(epsilonBase, VB_CountModel)
            tot_loss = loss + lamda * lossDF

            # zero the parameter gradients
            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()

        print('epoch: ', epoch, 'out of: ', num_epochs, 'average loss: ',
              tot_loss.item())
    return modelFair


# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:35:23 2020

@author: islam
"""
import pandas as pd
import numpy as np


# %%
# Measure intersectional DF from positive predict probabilities
def differentialFairnessBinaryOutcome(probabilitiesOfPositive):
    # input: probabilitiesOfPositive = positive p(y|S) from ML algorithm
    # output: epsilon = differential fairness measure
    epsilonPerGroup = np.zeros(len(probabilitiesOfPositive))
    for i in range(len(probabilitiesOfPositive)):
        epsilon = 0.0  # initialization of DF
        for j in range(len(probabilitiesOfPositive)):
            if i == j:
                continue
            else:
                epsilon = max(epsilon, abs(np.log(probabilitiesOfPositive[i]) - np.log(
                    probabilitiesOfPositive[
                        j])))  # ratio of probabilities of positive outcome
                epsilon = max(epsilon,
                              abs(np.log((1 - probabilitiesOfPositive[i])) - np.log((1 -
                                                                                     probabilitiesOfPositive[
                                                                                         j]))))  # ratio of probabilities of negative outcome
        epsilonPerGroup[i] = epsilon  # DF per group
    epsilon = max(epsilonPerGroup)  # overall DF of the algorithm
    return epsilon


# %%
# Measure SP-Subgroup fairness (gamma unfairness)
def subgroupFairness(probabilitiesOfPositive, alphaSP):
    # input: probabilitiesOfPositive = Pr[D(X)=1|g(x)=1]
    #        alphaG = Pr[g(x)=1]
    # output: gamma-unfairness
    spD = sum(
        probabilitiesOfPositive * alphaSP)  # probabilities of positive class across whole population SP(D) = Pr[D(X)=1]
    gammaPerGroup = np.zeros(len(probabilitiesOfPositive))  # SF per group
    for i in range(len(probabilitiesOfPositive)):
        gammaPerGroup[i] = alphaSP[i] * abs(spD - probabilitiesOfPositive[i])
    gamma = max(gammaPerGroup)  # overall SF of the algorithm
    return gamma


# %%
# mechanisms to make classification outcomes free of disparate impact, that is,
# to ensure that similar fractions of people from different demographic groups
# (e.g., males, females) are accepted (or classified as positive) by the classifier.
# More discussion about the disparate impact notion can be found in Sections 1 and 2 of the paper.
# paper: https://arxiv.org/pdf/1507.05259.pdf
def compute_p_rule(x_control, class_labels, predictProb):
    """ Compute the p-rule based on Doctrine of disparate impact """

    non_prot_all = sum(x_control == 1)  # non-protected group
    prot_all = sum(x_control == 0)  # protected group

    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter / numClasses

    non_prot_pos = sum(
        class_labels[x_control == 1] == 1)  # non_protected in positive class
    prot_pos = sum(class_labels[x_control == 0] == 1)  # protected in positive class
    frac_non_prot_pos = float(non_prot_pos) / float(non_prot_all)
    frac_prot_pos = float(prot_pos) / float(prot_all)
    p_rule = min(
        (frac_prot_pos + dirichletAlpha) / (frac_non_prot_pos + concentrationParameter),
        (frac_non_prot_pos + dirichletAlpha) / (
                frac_prot_pos + concentrationParameter)) * 100.0

    # soft p-rule
    non_prot_pos_soft = sum(
        predictProb[x_control == 1])  # non_protected in positive class
    prot_pos_soft = sum(predictProb[x_control == 0])  # protected in positive class
    frac_non_prot_pos_soft = float(non_prot_pos_soft) / float(non_prot_all)
    frac_prot_pos_soft = float(prot_pos_soft) / float(prot_all)
    p_rule_soft = min((frac_prot_pos_soft + dirichletAlpha) / (
            frac_non_prot_pos_soft + concentrationParameter),
                      (frac_non_prot_pos_soft + dirichletAlpha) / (
                              frac_prot_pos_soft + concentrationParameter)) * 100.0

    return p_rule, p_rule_soft


# %%
# smoothed empirical differential fairness measurement
def computeEDFforData(protectedAttributes, predictions, predictProb, intersectGroups):
    # compute counts and probabilities
    countsClassOne = np.zeros(len(intersectGroups))
    countsTotal = np.zeros(len(intersectGroups))
    countsClassOne_soft = np.zeros(len(intersectGroups))

    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter / numClasses
    population = len(predictions)

    # p-rule specific parameter
    x_control = np.int64(np.ones((len(predictions))))

    for i in range(len(predictions)):
        index = np.where((intersectGroups == protectedAttributes[i]).all(axis=1))[0][0]
        countsTotal[index] = countsTotal[index] + 1
        countsClassOne_soft[index] = countsClassOne_soft[index] + predictProb[i]
        if predictions[i] == 1:
            countsClassOne[index] = countsClassOne[index] + 1
        if protectedAttributes[i, 0] == 0 and protectedAttributes[i, 1] == 0 and \
                protectedAttributes[i, 2] == 0:
            x_control[i] = 0

    # probability of y given S (p(y=1|S)): probability distribution over merit per value of the protected attributes
    probabilitiesOfPositive_hard = (countsClassOne + dirichletAlpha) / (
            countsTotal + concentrationParameter)
    probabilitiesOfPositive_soft = (countsClassOne_soft + dirichletAlpha) / (
            countsTotal + concentrationParameter)
    alphaG_smoothed = (countsTotal + dirichletAlpha) / (
            population + concentrationParameter)

    epsilon_hard = differentialFairnessBinaryOutcome(probabilitiesOfPositive_hard)
    gamma_hard = subgroupFairness(probabilitiesOfPositive_hard, alphaG_smoothed)

    epsilon_soft = differentialFairnessBinaryOutcome(probabilitiesOfPositive_soft)
    gamma_soft = subgroupFairness(probabilitiesOfPositive_soft, alphaG_smoothed)

    p_rule, p_rule_soft = compute_p_rule(x_control, predictions, predictProb)
    return epsilon_hard, epsilon_soft, gamma_hard, gamma_soft, p_rule, p_rule_soft
