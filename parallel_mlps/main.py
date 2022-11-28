import logging
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from torchvision import datasets
from tqdm import tqdm

from parallel_mlps.parallel_mlp import ParallelMLPs

logger = logging.getLogger()

MIN_NEURONS = 1
MAX_NEURONS = 50
STEP = 1
REPETITIONS = 5
ACTIVATIONS = [nn.LeakyReLU(), nn.Sigmoid(), nn.ReLU(), nn.Identity()]
BATCH_SIZE = 256
NUM_EPOCHS = 50
DEVICE = "cuda"


def main():
    # Dataset creation
    transform = None
    mnist = datasets.MNIST("../data", train=True, download=True, transform=transform)
    x_train = mnist.data
    x_train = x_train.reshape((x_train.shape[0], -1)).float()
    y_train = mnist.targets

    mnist_test = datasets.MNIST(
        "../data", train=False, download=True, transform=transform
    )
    x_test = mnist_test.data
    x_test = x_test.reshape((x_test.shape[0], -1)).float()
    y_test = mnist_test.targets

    train_dataloader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=BATCH_SIZE
    )
    test_dataloader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE)

    # Model definition
    pmlps = ParallelMLPs(
        in_features=784,
        out_features=10,
        min_neurons=MIN_NEURONS,
        max_neurons=MAX_NEURONS,
        repetitions=REPETITIONS,
        step=STEP,
        activations=ACTIVATIONS,
        bias=True,
        device=DEVICE,
        logger=logger,
    )
    print(
        f"Created ParallelMLPs with {pmlps.num_unique_models}, starting with {MIN_NEURONS} neurons to {MAX_NEURONS} and step {STEP}, with activations {ACTIVATIONS}, repeated {REPETITIONS}."
    )

    optimizer = SGD(pmlps.parameters(), lr=1e-3)
    # Need to use reduction=none to make gradients independent in each subnetwork
    loss_function = nn.CrossEntropyLoss(reduction="none")

    # Training/Testing
    for _ in tqdm(range(NUM_EPOCHS)):
        train(pmlps, train_dataloader, loss_function, optimizer)
        test(pmlps, test_dataloader, loss_function)


def train(pmlps, train_dataloader, loss_function, optimizer):
    pmlps.train()
    gradient = torch.ones(pmlps.num_unique_models).to(DEVICE)

    for batch_idx, (x, y) in enumerate(train_dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = pmlps(x)  # [batch_size, num_unique_models, out_features]
        per_sample_candidate_losses = pmlps.calculate_loss(
            loss_func=loss_function, preds=outputs, target=y
        )  # [batch_size, num_unique_models]
        candidate_losses = per_sample_candidate_losses.mean(0)  # [num_unique_models]

        candidate_losses.backward(gradient=gradient)
        optimizer.step()


def test(pmlps, test_dataloader, loss_function):
    pmlps.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = pmlps(x)
            per_sample_candidate_losses = pmlps.calculate_loss(
                loss_func=loss_function, preds=outputs, target=y
            )

            candidate_losses = per_sample_candidate_losses.mean(0)
            test_loss += candidate_losses
            pred = outputs.argmax(
                dim=-1
            )  # [batch_size, num_unique_models, out_features] -> [batch_size, num_unique_models]
            correct += (pred == y[:, None]).sum(
                axis=0
            )  # [batch_size, num_unique_models] -> [num_unique_models]

    test_loss /= len(test_dataloader.dataset)
    candidates_acc = correct / len(test_dataloader.dataset)

    # Selecting best candidate
    best_candidate_id = candidates_acc.argmax().item()
    best_mlp = pmlps.extract_mlps([best_candidate_id])[0]

    tqdm.write(
        f"Test set: min candidate loss: {test_loss.min()}, max candidate acc: {candidates_acc.max()}, best mlp architecture: {best_mlp.short_repr()}"
    )
    return candidates_acc


if __name__ == "__main__":
    main()