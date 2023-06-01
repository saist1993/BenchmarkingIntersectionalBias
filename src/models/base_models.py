import torch
import torch.nn as nn
from torch.autograd import Function


class GradReverse(Function):
    """
        Torch function used to invert the sign of gradients (to be used for argmax instead of argmin)
        Usage:
            x = GradReverse.apply(x) where x is a tensor with grads.

        Copied from here: https://github.com/geraltofrivia/mytorch/blob/0ce7b23ff5381803698f6ca25bad1783d21afd1f/src/mytorch/utils/goodies.py#L39
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


class SimpleNonLinear(nn.Module):
    """Fairgrad uses this as complex non linear model"""

    def __init__(self, params):
        super().__init__()

        input_dim = params['model_arch']['encoder']['input_dim']
        output_dim = params['model_arch']['encoder']['output_dim']

        self.use_batch_norm = params['use_batch_norm']  # 0.0 means don't use batch norm
        self.use_dropout = params['use_dropout']  # 0.0 means don't use dropout

        self.layer_1 = nn.Linear(input_dim, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, 32)
        self.layer_out = nn.Linear(32, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.use_dropout)

        if self.use_batch_norm != 0.0:
            self.batchnorm1 = nn.InstanceNorm1d(128)
            self.batchnorm2 = nn.InstanceNorm1d(64)
            self.batchnorm3 = nn.InstanceNorm1d(32)

    def forward(self, params):
        x = params['input']
        x = self.layer_1(x)
        if self.use_batch_norm:
            x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)  # This does not exists in Michael.

        x = self.layer_2(x)
        if self.use_batch_norm:
            x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        z = self.layer_3(x)
        if self.use_batch_norm:
            z = self.batchnorm3(z)
        z = self.relu(z)
        z = self.dropout(z)

        z = self.layer_out(z)

        output = {
            'prediction': z,
            'adv_output': None,
            'hidden': x,  # just for compatabilit
            'classifier_hiddens': None,
            'adv_hiddens': None
        }

        return output

    @property
    def layers(self):
        return torch.nn.ModuleList([self.layer_1, self.layer_2, self.layer_3, self.layer_out])


class AdversarialSingle(nn.Module):
    def __init__(self, params):
        super().__init__()

        input_dim = params['model_arch']['encoder']['input_dim']
        output_dim = params['model_arch']['encoder']['output_dim']
        # Generalizing this to n-adversarial
        adv_dims = params['model_arch']['adv']['output_dim']  # List with n adversarial output!

        dropout = self.use_dropout

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.adversarials = []
        for adv_dim in adv_dims:
            self.adversarials.append(nn.Sequential(
                nn.Linear(128, 64),
                # nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(64, 32),
                # nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(32, adv_dim)
            ))

        self.adversarials = nn.ModuleList(self.adversarials)

        self.task_classifier = nn.Sequential(
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, output_dim)
        )

    def forward(self, params):
        x = params['input']
        encoder_output = self.encoder(x)
        adv_output = [adv(GradReverse.apply(encoder_output)) for adv in self.adversarials]
        task_classifier_output = self.task_classifier(encoder_output)

        output = {
            'prediction': task_classifier_output,
            'adv_outputs': adv_output,
            'hidden': encoder_output,  # just for compatabilit
            'classifier_hiddens': None,
            'adv_hiddens': None
        }

        return output

    @property
    def layers(self):
        return torch.nn.ModuleList([self.encoder, self.adversarials, self.task_classifier])
