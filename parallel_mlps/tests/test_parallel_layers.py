import logging
import os
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import SGD

from parallel_mlps.parallel_mlp import ParallelMLPs

logger = logging.getLogger()

N_SAMPLES = 5
N_FEATURES = 3
N_OUTPUTS = 2

MIN_NEURONS = 1
MAX_NEURONS = 3


def reproducibility():
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture
def X():
    return torch.rand(size=(N_SAMPLES, N_FEATURES))


@pytest.fixture
def Y():
    return torch.randint(low=0, high=2, size=(N_SAMPLES,))


@pytest.fixture
def activation_functions():
    return [nn.LeakyReLU(), nn.Sigmoid()]


@pytest.fixture
def parallel_mlp_object(activation_functions, X):
    return ParallelMLPs(
        in_features=X.shape[1],
        out_features=N_OUTPUTS,
        min_neurons=MIN_NEURONS,
        max_neurons=MAX_NEURONS,
        repetitions=3,
        step=1,
        activations=activation_functions,
        bias=True,
        device="cpu",
        logger=logger,
    )


def test_trainings(X, Y, parallel_mlp_object):
    reproducibility()
    lr = 0.5
    atol = 1e-5
    rtol = 1e-5
    parallel_optimizer = SGD(params=parallel_mlp_object.parameters(), lr=lr)

    single_models = [
        parallel_mlp_object.extract_mlps([i])[0]
        for i in parallel_mlp_object.unique_model_ids
    ]
    single_optimizers = [
        SGD(params=model.parameters(), lr=lr) for model in single_models
    ]

    num_epochs = 100
    parallel_loss = nn.CrossEntropyLoss(reduction="none")
    sequential_loss = nn.CrossEntropyLoss()

    X = X.to(parallel_mlp_object.device)
    Y = Y.to(parallel_mlp_object.device)
    gradient = torch.ones(parallel_mlp_object.num_unique_models).to(X.device)

    for e in range(num_epochs):
        print(f"Epoch: {e}")
        parallel_optimizer.zero_grad()
        outputs = parallel_mlp_object(X)
        per_sample_candidate_losses = parallel_mlp_object.calculate_loss(
            parallel_loss, outputs, Y
        )
        candidate_losses = per_sample_candidate_losses.mean(0)
        candidate_losses.backward(gradient=gradient)
        parallel_optimizer.step()

        for i, (model, optimizer) in enumerate(zip(single_models, single_optimizers)):
            optimizer.zero_grad()
            single_outputs = model(X)
            loss = sequential_loss(single_outputs, Y)
            loss.backward()
            optimizer.step()

            # Asserts
            assert torch.allclose(candidate_losses[i], loss, atol=atol, rtol=rtol)

            m = parallel_mlp_object.extract_mlps([i])[0]
            print(i)

            np.testing.assert_allclose(
                m.hidden_layer.weight.detach().cpu().numpy(),
                model.hidden_layer.weight.detach().cpu().numpy(),
                atol=atol,
                rtol=rtol,
            )
            np.testing.assert_allclose(
                m.hidden_layer.bias.detach().cpu().numpy(),
                model.hidden_layer.bias.detach().cpu().numpy(),
                atol=atol,
                rtol=rtol,
            )
            np.testing.assert_allclose(
                m.out_layer.weight.detach().cpu().numpy(),
                model.out_layer.weight.detach().cpu().numpy(),
                atol=atol,
                rtol=rtol,
            )
            np.testing.assert_allclose(
                m.out_layer.bias.detach().cpu().numpy(),
                model.out_layer.bias.detach().cpu().numpy(),
                atol=atol,
                rtol=rtol,
            )
            assert m.activation == model.activation
            assert type(m) == type(model)

        if (e == 0) or (e == (num_epochs - 1)):
            print(f"Epoch: {e}")
            print(m.hidden_layer.weight)
            print(model.hidden_layer.weight)

            print(m.hidden_layer.bias)
            print(model.hidden_layer.bias)

            print(m.out_layer.weight)
            print(model.out_layer.weight)

            print(m.out_layer.bias)
            print(model.out_layer.bias)