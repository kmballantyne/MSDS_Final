import copy
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from config import *
from server import average_weights  # might be useful if you extend later

def copy_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)

def ditto_local_train(
    global_model: nn.Module,
    personal_model: nn.Module,
    dataset,
    mu: float = DITTO_MU,
    local_epochs: int = LOCAL_EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
):
    """
    Perform Ditto local training on a single client.

    Args:
        global_model: nn.Module, initialized with current global weights
        personal_model: nn.Module, initialized (usually) with same global weights or prior personalized weights
        dataset: client's Dataset
        mu: proximal regularization strength (Ditto's lambda)
        local_epochs: number of local epochs
        lr: learning rate
        batch_size: local batch size

    Returns:
        new_global_state: state_dict for global model update
        new_personal_state: state_dict for personalized model
    """
    global_model = global_model.to(DEVICE)
    personal_model = personal_model.to(DEVICE)
    global_model.train()
    personal_model.train()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    opt_g = optim.SGD(global_model.parameters(), lr=lr, momentum=0.9)
    opt_p = optim.SGD(personal_model.parameters(), lr=lr, momentum=0.9)

    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # --- Update global model: plain CE (FedAvg-style) ---
            opt_g.zero_grad()
            logits_g = global_model(x)
            loss_g = criterion(logits_g, y)
            loss_g.backward()
            opt_g.step()

            # --- Update personal model: CE + proximal term ---
            opt_p.zero_grad()
            logits_p = personal_model(x)
            ce_p = criterion(logits_p, y)

            prox = 0.0
            # use current global_model params as anchor; detach to avoid gradients into global
            for (w_p, w_g) in zip(personal_model.parameters(), global_model.parameters()):
                prox = prox + torch.sum((w_p - w_g.detach()) ** 2)

            loss_p = ce_p + 0.5 * mu * prox
            loss_p.backward()
            opt_p.step()

    return global_model.state_dict(), personal_model.state_dict()
