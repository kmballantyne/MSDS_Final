
import copy
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from config import * # expects DEVICE at least
from server import average_weights
# from model import get_model

def copy_model(model: nn.Module) -> nn.Module:
    """
    Creates a copy of the given model with the same weights.
    """
    return copy.deepcopy(model)


def personalize(
    model: nn.Module, 
    dataset: torch.utils.data.Dataset, 
    steps: int = 5, 
    lr: float = 0.001,
    batch_size: int = BATCH_SIZE,
) -> nn.Module:
    """
    Fine-tunes a copy of the model on the client's data across 'steps' for gradient steps.
    
    Args:
        model: Global model to start from.
        dataset: Client dataset (torch.utils.data.Dataset).
        steps: Number of optimization steps to run.
        lr: Learning rate for personalization.
        batch_size: Minibatch size.
    
    Returns: 
        A personalized model (deep-copied from 'model').
    """
    # Copy the global model so we don't mutate it in-place
    model_copy = copy_model(model).to(DEVICE)
    model_copy.train()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_copy.parameters(), lr=lr, momentum=0.9)
    
    step_count = 0
    while step_count < steps:
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model_copy(x) # forward pass; shape [batch_size, num_classes]
            loss = criterion(logits, y) # y: 0/1 labels for binary classification
            loss.backward()
            optimizer.step()
            
            step_count += 1
            if step_count >= steps:
                break
    
    return model_copy
    
    # for _ in range(steps):
    #     for x, y in loader:
    #         x, y = x.to(DEVICE), y.to(DEVICE)
    #         optimizer.zero_grad()
    #         output = model_copy(x)
    #         loss = criterion(output, y)
    #         loss.backward()
    #         optimizer.step()
    # return model_copy

# def copy_model(model):
#     new_model = get_model()
#     new_model.load_state_dict(model.state_dict())
#     return new_model.to(DEVICE)

def per_fedavg_train(
    global_model: nn.Module, 
    client_datasets: list[torch.utils.data.Dataset], 
    personalization_steps: int = 5,
    lr: float = 0.001,
    batch_size: int = BATCH_SIZE,
):
    """
    Performs a Per-FedAvg-style training round.

    For each client:
      - Start from the current global model.
      - Run `personalization_steps` of local fine-tuning (personalize).
    Then:
      - Average the personalized weights to form the new global state.

    Args:
        global_model: Current global model (nn.Module).
        client_datasets: List of client datasets.
        personalization_steps: Number of local steps per client.
        lr: Learning rate for personalization.
        batch_size: Minibatch size for personalization.

    Returns:
        new_global_state: state_dict of the updated global model.
        personalized_states: list of state_dicts, one per client.
    """
    global_model.to(DEVICE)
    global_model.eval()
    
    personalized_states = []
    
    for dataset in client_datasets:
        personalized_model = personalize(
            global_model, 
            dataset, 
            steps=personalization_steps, 
            lr=lr, 
            batch_size=batch_size
        )
        personalized_states.append(personalized_model.state_dict())
        
    # Aggregate personalized weights to update global model
    new_global_state = average_weights(personalized_states)
    global_model.load_state_dict(new_global_state)

    return new_global_state, personalized_states