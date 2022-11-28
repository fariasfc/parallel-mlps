import logging
import math
from copy import deepcopy
from functools import partial
from itertools import groupby
from typing import Any, Counter, List

import numpy as np
import torch
from joblib import Parallel, delayed
from torch import nn, random
from torch._C import Value
from torch.functional import Tensor
from torch.multiprocessing import Pool, freeze_support, set_start_method
from torch.nn import init
from torch.nn.modules.linear import Linear
from torch.nn.parameter import Parameter

MAP_ACTIVATION = {
    "sigmoid": nn.Sigmoid,
    "Sigmoid()": nn.Sigmoid,
    "relu": nn.ReLU,
    "ReLU()": nn.ReLU,
    "tanh": nn.Tanh,
    "Tanh()": nn.Tanh,
    "selu": nn.SELU,
    "SELU()": nn.SELU,
    "leakyrelu": nn.LeakyReLU,
    "LeakyReLU(negative_slope=0.01)": nn.LeakyReLU,
    "identity": nn.Identity,
    "Identity()": nn.Identity,
    "elu": nn.ELU,
    "ELU()": nn.ELU,
    "gelu": nn.GELU,
    "GELU()": nn.GELU,
    "mish": nn.Mish,
    "Mish()": nn.Mish,
    "hardshrink": nn.Hardshrink,
    "Hardshrink()": nn.Hardshrink,
}


class MLP(nn.Module):
    def __init__(self, hidden_layer, out_layer, activation, model_id, metadata, device):
        super().__init__()
        self.hidden_layer = hidden_layer
        self.out_layer = out_layer
        self.activation = activation
        self.model_id = model_id
        self.metadata = metadata
        self.device = device

    def __str__(self):
        s = "MLP(\n"

        if self.hidden_layer:
            s += f"(hidden_layer): {self.hidden_layer}\n"

        if self.activation:
            s += f"(activation): {self.activation}\n"

        if self.out_layer:
            s += f"(out_layer): {self.out_layer}\n"

        s += ")"

        return s

    def __repr__(self):
        return self.__str__()

    def short_repr(self):
        return f"{self.hidden_layer.in_features}-{self.hidden_layer.out_features}-{self.activation}-{self.out_layer.out_features}"

    @property
    def out_features(self):
        if self.out_layer is not None:
            return self.out_layer.out_features
        else:
            return self.hidden_layer.out_features

    def forward(self, x):
        if self.hidden_layer is not None:
            x = self.hidden_layer(x)

        if self.activation is not None:
            x = self.activation(x)

        if self.out_layer is not None:
            x = self.out_layer(x)

        return x


def build_model_ids(
    repetitions: int,
    activation_functions: list,
    min_neurons: int,
    max_neurons: int,
    step: int,
):
    """Creates a list with model ids to relate to hidden representations.
    1. Creates a list containing the number of hidden neurons for each architecture (independent of activation functions and/or repetitions)
    using the following formula neurons_structures=range(min_neurons, max_neurons+1, step)
    2. Calculates the number of independent models (parallel mlps) = len(neurons_structures) * len(activations) * repetitions

    Raises:
        ValueError: [description]
        ValueError: [description]
        RuntimeError: [description]
        ValueError: [description]

    Returns:
        hidden_neurons__model_id: List indicating for each global neuron the model_id that it belongs to
        output__model_id: List containing the id of the model for each output
        output__architecture_id: List containing the id of the architecture (neuron structure AND activation function) that the output belongs.
            Architectures with the same id means that it only differs the repetition number, but have equal neuron structure and activation function.
    """

    if len(activation_functions) == 0:
        raise ValueError(
            "At least one activation function must be passed. Try `nn.Identity()` if you want no activation."
        )

    activation_names = [a.__class__.__name__ for a in activation_functions]
    if len(set(activation_names)) != len(activation_names):
        raise ValueError("activation_functions must have only unique values.")

    neurons_structure = torch.arange(min_neurons, max_neurons + 1, step).tolist()

    num_activations = len(activation_functions)
    num_different_neurons_structures = len(neurons_structure)
    num_parallel_mlps = num_different_neurons_structures * num_activations * repetitions

    i = 0
    hidden_neuron__model_id = []
    while i < num_parallel_mlps:
        for structure in neurons_structure:
            hidden_neuron__model_id += [i] * structure
            i += 1

    total_hidden_neurons = len(hidden_neuron__model_id)
    activations_split = total_hidden_neurons // num_activations

    output__model_id = torch.Tensor(
        [i[0] for i in groupby(hidden_neuron__model_id)]
    ).long()
    output__neuron_structure_id = (
        output__model_id[: num_activations * num_different_neurons_structures]
        * repetitions
    )

    repetition_architecture_id = (
        torch.arange(num_different_neurons_structures).repeat(repetitions).long()
    )
    # # model_ids = np.arange(num_different_neurons_structures)
    # repetition_architecture_id = np.array([])
    # for rep in range(repetitions):
    #     repetition_architecture_id = np.hstack((repetition_architecture_id, model_ids))

    output__architecture_id = torch.tensor([])
    increment = max(repetition_architecture_id) + 1
    for act in range(num_activations):
        output__architecture_id = torch.hstack(
            (output__architecture_id, repetition_architecture_id.clone())
        )
        repetition_architecture_id += increment
        # [0, 1, 0, 1]
        # [2, 3, 2, 3] + 2
        # [] + 4

    output__repetition = (
        torch.arange(repetitions)
        .repeat_interleave(num_different_neurons_structures)
        .repeat(num_activations)
    ).long()

    output__architecture_id = output__architecture_id.long()
    hidden_neuron__model_id = torch.Tensor(hidden_neuron__model_id).long()

    assert len(output__architecture_id) == len(output__model_id)
    assert len(output__architecture_id) == len(output__repetition)

    return (
        hidden_neuron__model_id,
        output__model_id,
        output__architecture_id,
        output__repetition,
    )


class ParallelMLPs(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        min_neurons: int,
        max_neurons: int,
        step: int,
        repetitions: int,
        activations: List[nn.Module],
        bias: bool = True,
        device: str = "cuda",
        logger: Any = None,
    ):
        super().__init__()

        (
            hidden_neuron__model_id,
            output__model_id,
            output__architecture_id,
            output__repetition,
        ) = build_model_ids(
            repetitions=repetitions,
            activation_functions=activations,
            min_neurons=min_neurons,
            max_neurons=max_neurons,
            step=step,
        )

        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.activations = activations
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

        # Mappings: index -> id
        self.hidden_neuron__model_id = hidden_neuron__model_id.to(self.device)

        self.output__model_id = output__model_id.to(self.device)
        self.output__architecture_id = output__architecture_id.to(self.device)
        self.output__repetition = output__repetition.to(self.device)

        self.model_id__hidden_idx = {i.item(): [] for i in self.output__model_id.cpu()}
        for i, ix in enumerate(hidden_neuron__model_id):
            self.model_id__hidden_idx[ix.item()].append(i)

        # self.model_id__hidden_idxs = {
        #     i: torch.where(self.hidden_neuron__model_id)[0]
        #     for i in self.output__model_id
        # }
        self.hidden_neuron__model_id = self.hidden_neuron__model_id.to(self.device)
        # self.output__repetition = (
        #     torch.Tensor(output__repetition.float()uuuuuuuu).long().to(self.device)
        # )

        self.total_hidden_neurons = len(self.hidden_neuron__model_id)
        self.unique_model_ids = sorted(list(set(hidden_neuron__model_id.tolist())))
        self.model_id__num_hidden_neurons = torch.from_numpy(
            np.bincount(self.hidden_neuron__model_id.cpu().numpy())
        ).to(self.device)
        # self.model_id__start_neuron = torch.zeros_like(self.model_id__num_hidden_neurons)
        self.model_id__start_idx = torch.cat(
            [
                torch.tensor([0]).to(self.device),
                self.model_id__num_hidden_neurons.cumsum(0)[:-1],
            ]
        )
        self.model_id__end_idx = (
            self.model_id__start_idx + self.model_id__num_hidden_neurons
        )
        # self.model_id__start_neuron =

        # self.model_id__num_hidden_neurons = torch.bincount(
        #     self.hidden_neuron__model_id
        # ).to(self.device)

        self.num_unique_models = len(self.unique_model_ids)
        self.num_activations = len(activations)

        self.activations_split = self.total_hidden_neurons // self.num_activations

        self.model_id__activation_id = (
            self.model_id__num_hidden_neurons.cumsum(0) - 1
        ) // self.activations_split
        self.model_id__activation_id = self.model_id__activation_id.cpu()
        # # Adjusting because the last element is always increased due to the cumsum.
        # self.model_id__activation_id[-1] -= 1

        self.hidden_layer = nn.Linear(self.in_features, self.total_hidden_neurons)
        self.weight = Parameter(
            torch.Tensor(self.out_features, self.total_hidden_neurons)
        )
        if bias:
            self.bias = Parameter(
                torch.Tensor(self.num_unique_models, self.out_features)
            )
        else:
            self.bias = None
            self.register_parameter("bias", None)

        self.reset_parameters()
        self.to(device)
        self.logger.info(f"Model sent to {device}!")

    def _build_outputs_ids(self):
        return [i[0] for i in groupby(self.hidden_neuron__model_id)]

    def reset_parameters(self, layer_ids=None):
        if layer_ids == None:
            layer_ids = self.unique_model_ids

        with torch.no_grad():
            for layer_id in layer_ids:
                start = self.model_id__start_idx[layer_id]
                end = self.model_id__end_idx[layer_id]
                hidden_w = self.hidden_layer.weight[start:end, :]
                hidden_b = self.hidden_layer.bias[start:end]

                out_w = self.weight[:, start:end]
                out_b = self.bias[layer_id, :]

                for w, b in [(hidden_w, hidden_b), (out_w, out_b)]:
                    init.kaiming_uniform_(w, a=math.sqrt(5))
                    fan_in, _ = init._calculate_fan_in_and_fan_out(w)
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(b, -bound, bound)

    def apply_activations(self, x: Tensor) -> Tensor:
        tensors = x.split(self.activations_split, dim=1)
        output = []
        sub_tensor_out_features = tensors[0].shape[1]
        for (act, sub_tensor) in zip(self.activations, tensors):
            if sub_tensor.shape[1] != sub_tensor_out_features:
                raise RuntimeError(
                    f"sub_tensors with different number of parameters per activation {[t.shape for t in tensors]}"
                )
            output.append(act(sub_tensor))
        output = torch.cat(output, dim=1)
        return output

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        x = self.hidden_layer(x)  # [batch_size, total_hidden_neurons]
        x = self.apply_activations(x)  # [batch_size, total_hidden_neurons]

        x = (
            x[:, :, None] * self.weight.T[None, :, :]
        )  # [batch_size, total_hidden_neurons, 1] * [1, total_hidden_neurons, out_features] = [batch_size, total_hidden_neurons, out_features]

        # [batch_size, total_repetitions, num_architectures, out_features]
        adjusted_out = (
            torch.zeros(
                batch_size, self.num_unique_models, self.out_features, device=x.device
            ).scatter_add_(
                1,
                self.hidden_neuron__model_id[None, :, None].expand(
                    batch_size, -1, self.out_features
                ),  # [batch_size, total_hidden_neurons, out_features]. expand does not consumes memory.
                x,
            )
        ) + self.bias[None, :, :]

        # [batch_size, num_unique_models, out_features]
        return adjusted_out

    def calculate_loss(self, loss_func, preds, target):
        if hasattr(loss_func, "reduction"):
            if loss_func.reduction != "none":
                raise ValueError(
                    "Loss function for ParallelMLPs must have reduction=none"
                )

        if preds.ndim == 3:
            batch_size, num_models, neurons = preds.shape
            loss = loss_func(
                preds.permute(0, 2, 1), target[:, None].expand(-1, num_models)
            )
        else:
            loss = loss_func(preds, target)

        return loss

    def extract_params_mlps(self, model_ids: List[int]) -> List[MLP]:
        """Extracts a completely independent MLP."""
        if max(model_ids) >= self.num_unique_models:
            raise ValueError(
                f"model_id {max(model_ids)} > num_uniqe_models {self.num_unique_models}"
            )

        rets = []
        with torch.no_grad():
            for model_id in model_ids:
                # model_neurons = torch.where(self.hidden_neuron__model_id == model_id)[0]
                model_neurons = self.model_id__hidden_idx[model_id]
                hidden_weight = self.hidden_layer.weight[model_neurons, :]
                hidden_bias = self.hidden_layer.bias[model_neurons]

                out_weight = self.weight[:, model_neurons]
                out_bias = self.bias[model_id, :]

                activation = self.get_activation_from_model_id(model_id)

                ret = {
                    "hidden_weight": hidden_weight.clone(),
                    "hidden_bias": hidden_bias.clone(),
                    "out_weight": out_weight.clone(),
                    "out_bias": out_bias.clone(),
                    "activation": activation,
                    "metadata": {"model_id": model_id},
                }
                rets.append(ret)

                # mlps.append(
                #     MLP(
                #         hidden_layer=hidden_layer,
                #         out_layer=out_layer,
                #         activation=activation,
                #         model_id=model_id,
                #         metadata={"model_id": model_id},
                #         device=self.device,
                #     )
                # )

        return rets

    # @profile
    def extract_mlps(self, model_ids: List[int]) -> List[MLP]:
        """Extracts a completely independent MLP."""
        if max(model_ids) >= self.num_unique_models:
            raise ValueError(
                f"model_id {max(model_ids)} > num_uniqe_models {self.num_unique_models}"
            )

        mlps = []
        with torch.no_grad():
            for model_id in model_ids:
                model_neurons = torch.where(self.hidden_neuron__model_id == model_id)[0]
                # model_neurons = self.model_id__hidden_idx[model_id]
                hidden_weight = self.hidden_layer.weight[model_neurons, :]
                hidden_bias = self.hidden_layer.bias[model_neurons]

                out_weight = self.weight[:, model_neurons]
                out_bias = self.bias[model_id, :]

                hidden_layer = nn.Linear(
                    in_features=hidden_weight.shape[1],
                    out_features=hidden_weight.shape[0],
                )
                activation = self.get_activation_from_model_id(model_id)
                out_layer = nn.Linear(
                    in_features=hidden_layer.out_features,
                    out_features=self.out_features,
                )

                hidden_layer.weight[:, :] = hidden_weight
                hidden_layer.bias[:] = hidden_bias

                out_layer.weight[:, :] = out_weight
                out_layer.bias[:] = out_bias

                mlps.append(
                    MLP(
                        hidden_layer=hidden_layer,
                        out_layer=out_layer,
                        activation=activation,
                        model_id=model_id,
                        metadata={"model_id": model_id},
                        device=self.device,
                    )
                )

        return mlps

    def get_regularization_term(self, gamma=1e-4, l=2):
        n_hidden, n_inputs = self.hidden_layer.weight.shape
        n_models = self.num_unique_models

        if l == 1:
            hid_w = (self.hidden_layer.weight.abs()).sum(-1)[:, None]
            hid_b = (self.hidden_layer.bias.abs())[:, None]
            out_w = (self.weight.abs()).sum(0)[:, None]
            b_hid_out_reg = (self.bias.abs()).sum(-1)
        else:
            hid_w = (self.hidden_layer.weight ** l).sum(-1)[:, None]
            hid_b = (self.hidden_layer.bias ** l)[:, None]
            out_w = (self.weight ** l).sum(0)[:, None]
            b_hid_out_reg = (self.bias ** l).sum(-1)

        w_in_hid_reg = torch.zeros(n_models, 1, device=self.device).scatter_add_(
            0,
            self.hidden_neuron__model_id[:, None],
            hid_w,
        )

        b_in_hid_reg = torch.zeros(n_models, 1, device=self.device).scatter_add(
            0,
            self.hidden_neuron__model_id[:, None],
            hid_b,
        )

        w_hid_out_reg = torch.zeros(n_models, 1, device=self.device).scatter_add_(
            0,
            self.hidden_neuron__model_id[:, None],
            out_w,
        )

        reg = w_in_hid_reg + b_in_hid_reg + w_hid_out_reg + b_hid_out_reg[:, None]
        reg = reg.flatten() / self.model_id__num_hidden_neurons
        reg = gamma * reg

        return reg

    def get_model_ids_from_architecture_id(self, architecture_id):
        indexes = self.output__architecture_id == architecture_id
        model_ids = self.output__model_id[indexes]
        return model_ids.cpu().tolist()

    def get_num_hidden_neurons_from_architecture_id(self, architecture_id):
        model_id = self.get_model_ids_from_architecture_id(architecture_id)[0]
        return self.get_num_hidden_neurons_from_model_id(model_id)

    def get_num_hidden_neurons_from_model_id(self, model_id):
        return ((self.hidden_neuron__model_id == model_id)).sum()

    def get_architecture_ids_from_model_ids(self, model_ids):
        architecture_id = self.output__architecture_id[model_ids]
        return architecture_id.cpu().tolist()

    def get_activation_from_model_id(self, model_ids):
        activations = []
        if isinstance(model_ids, (int, np.int64)):
            model_ids = [model_ids]

        for model_id in model_ids:
            activations.append(self.activations[self.model_id__activation_id[model_id]])

        if len(activations) == 1:
            activations = activations[0]

        return activations

    def old_get_activation_from_model_id(self, model_ids):
        activations = []
        if isinstance(model_ids, int):
            model_ids = [model_ids]

        for model_id in model_ids:
            activation_index = (
                torch.nonzero(self.hidden_neuron__model_id == model_id)[0]
                // self.activations_split
            )
            activation = deepcopy(self.activations[activation_index])
            activations.append(activation)
        if len(activations) == 1:
            activations = activations[0]

        return activations

    def get_activation_name_from_model_id(self, model_id):
        activation = self.get_activation_from_model_id(model_id)
        activation_name = [
            k for (k, v) in MAP_ACTIVATION.items() if v == type(activation)
        ][0]
        return activation_name

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
