import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Callable, Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [1, 10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)


class SimpleModelGeneratorComplex(nn.Module):
    """Fairgrad uses this as complex non linear model"""

    def __init__(self, input_dim, number_of_params=None):
        super().__init__()

        self.layer_1 = nn.Linear(input_dim, input_dim, bias=False)
        # self.layer_2 = nn.Linear(125, 50)
        # self.layer_3 = nn.Linear(50, input_dim)
        # self.leaky_relu = nn.LeakyReLU()

        # if number_of_params == 3:
        #     self.lambda_params = torch.nn.Parameter(torch.FloatTensor([0.33, 0.33, 0.33]))
        # elif number_of_params == 4:
        #     self.lambda_params = torch.nn.Parameter(torch.FloatTensor([0.25, 0.25, 0.25, 0.25]))

    def forward(self, other_examples):
        final_output = torch.tensor(0.0, requires_grad=True)
        for group in other_examples:
            x = group['input']
            x = self.layer_1(x)
            # x = self.leaky_relu(x)
            # x = self.layer_2(x)
            # x = self.leaky_relu(x)
            # x = self.layer_3(x)
            final_output = final_output + x

        output = {
            'prediction': final_output,
            'adv_output': None,
            'hidden': x,  # just for compatabilit
            'classifier_hiddens': None,
            'adv_hiddens': None
        }

        return output

    @property
    def layers(self):
        return torch.nn.ModuleList([self.layer_1, self.layer_2, self.layer_3])


class SimpleModelGenerator(nn.Module):
    """Fairgrad uses this as complex non linear model"""

    def __init__(self, input_dim, number_of_params=3):
        super().__init__()

        if number_of_params == 3:
            self.lambda_params = torch.nn.Parameter(torch.FloatTensor([0.33, 0.33, 0.33]))
        elif number_of_params == 4:
            self.lambda_params = torch.nn.Parameter(torch.FloatTensor([0.25, 0.25, 0.25, 0.25]))

        # self.more_lambda_params = nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(torch.ones(input_dim))) for i in
        #                            range(len(self.lambda_params))])

        # self.more_lambda_params = torch.nn.Parameter(torch.FloatTensor(torch.ones(input_dim)))

    def forward(self, other_examples):
        final_output = torch.tensor(0.0, requires_grad=True)
        for param, group in zip(self.lambda_params, other_examples):
            x = group['input']
            final_output = final_output + x * param

        output = {
            'prediction': final_output,
            'adv_output': None,
            'hidden': x,  # just for compatabilit
            'classifier_hiddens': None,
            'adv_hiddens': None
        }

        return output

    @property
    def layers(self):
        return torch.nn.ModuleList([self.layer_1, self.layer_2])


class SimpleModelGeneratorIntermediate(nn.Module):
    """Fairgrad uses this as complex non linear model"""

    def __init__(self, input_dim, number_of_params=3):
        super().__init__()

        if number_of_params == 3:
            self.lambda_params = torch.nn.Parameter(torch.FloatTensor([0.33, 0.33, 0.33]))
        elif number_of_params == 4:
            self.lambda_params = torch.nn.Parameter(torch.FloatTensor([0.1, 0.1, 0.1, 0.1]))

        # self.more_lambda_params = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.more_lambda_params = torch.nn.Parameter(torch.FloatTensor(torch.randn(input_dim)))
        # nn.init.constant_(self.more_lambda_params.weight, 1.0)
        print("here")
        # self.more_lambda_params = [torch.nn.init.orthogonal_(l.reshape(1,-1)).squeeze() for l in self.more_lambda_params]

        # self.more_lambda_params = torch.nn.Parameter(torch.FloatTensor(torch.ones(input_dim)))

    def forward(self, other_examples):
        # final_output = torch.tensor(0.0, requires_grad=True)

        input = torch.sum(torch.stack([i['input'] for i in other_examples]), axis=0)
        # final_output = self.more_lambda_params(input)
        final_output = self.more_lambda_params * input
        # for param, group in zip(self.more_lambda_params, other_examples):
        #     x = group['input']
        #     final_output = final_output + param(x)

        output = {
            'prediction': final_output,
            'adv_output': None,
            'hidden': input,  # just for compatability
            'classifier_hiddens': None,
            'adv_hiddens': None
        }

        return output

    @property
    def layers(self):
        return torch.nn.ModuleList([self.layer_1, self.layer_2])
