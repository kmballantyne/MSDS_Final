from model import get_model
from client import local_train
from server import average_weights
from utils import load_client_datasets
from config import *
import torch

from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import BinaryAUROC 
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryAveragePrecision

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter

def main():
    print("[DEBUG] Entered main()")
    global_model = get_model().to(DEVICE)
    print("[DEBUG] Model created on", DEVICE)
    clients_data = load_client_datasets()
    print("[DEBUG] Client datasets loaded")
    
    writer = SummaryWriter(log_dir="runs/fedavg_experiment")
    
    # Initialize the global model
    for round in range(NUM_ROUNDS):
        print(f"--- Federated Round {round + 1}/{NUM_ROUNDS} ---")
        client_weights = []
        
        # Train each client's model
        for i in range(NUM_CLIENTS):
            print(f" --- Training on client {i + 1}/{NUM_CLIENTS} ---")
            
            client_model = get_model().to(DEVICE)
            client_model.load_state_dict(global_model.state_dict())
            w = local_train(client_model, clients_data[i])
            client_weights.append(w)
            
            print(f"--- Client {i + 1} trained successfully. ---")
            
        print(f"--- All clients trained. Aggregating weights... ---")
        # Average the weights from all clients
        global_weights = average_weights(client_weights)
        
        # Global model update
        global_model.load_state_dict(global_weights)
        print(f"--- Global model updated with averaged weights. ---")
        
        # Global model evaluation
        combined_dataset = ConcatDataset(clients_data)
        global_metrics = evaluate(global_model, combined_dataset)
        print(
            f"Round {round + 1} - "
            f"Accuracy: {global_metrics['accuracy']:.4f}, "
            f"Recall: {global_metrics['recall']:.4f}, "
            f"AUPRC: {global_metrics['auprc']:.4f}"
        )
        # print(f"Round {round + 1} - Accuracy: {global_metrics['accuracy']:.4f}, Recall: {global_metrics['recall']:.4f}, AUPRC: {global_metrics['auprc']:.4f}")
        
        # Log metrics to TensorBoard
        writer.add_scalar("Global/Accuracy", global_metrics['accuracy'], round)
        writer.add_scalar("Global/Recall", global_metrics['recall'], round)
        writer.add_scalar("Global/AUPRC", global_metrics['auprc'], round)
        
        # Log to CSV (global model metrics)
        with open("global_metrics.csv", "a") as f:
            if round == 0:
                f.write("round,accuracy,recall,auprc\n")
            f.write(
                f"{round+1},"
                f"{global_metrics['accuracy']:.4f}," 
                f"{global_metrics['recall']:.4f},"
                f"{global_metrics['auprc']:.4f}\n"
            )
        
        # Log to CSV (per-client metrics)
        with open("per_client_metrics.csv", "a") as f:
            if round == 0:
                f.write("round,client,accuracy,recall,auprc\n")
            for i, dataset in enumerate(clients_data):
                metrics = evaluate(global_model, dataset)
                f.write(
                    f"{round+1},client_{i},"
                    f"{metrics['accuracy']:.4f},"
                    f"{metrics['recall']:.4f},"
                    f"{metrics['auprc']:.4f}\n"
                )
                # f.write(f"{round+1},client_{i},{metrics['accuracy']:.4f},{metrics['recall']:.4f},{metrics['auprc']:.4f}\n")
                writer.add_scalar(f"Client_{i}/Accuracy", metrics['accuracy'], round)

        print(f"--- Global model updated after round {round + 1}. ---")
        
    # Optional: Save model or run final evaluation
    torch.save(global_model.state_dict(), "fedavg_global_model.pth")
    writer.close()
    
def evaluate(model, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=32) # or EVAL_BATCH_SIZE from config
    
    metric_acc = BinaryAccuracy().to(DEVICE)
    metric_recall = BinaryRecall().to(DEVICE)
    metric_auprc = BinaryAveragePrecision().to(DEVICE)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            probs = torch.softmax(preds, dim=1)[:, 1]  # Get class 1 probabilities
            preds_class = torch.argmax(preds, dim=1)

            metric_acc.update(preds_class, y)
            metric_recall.update(preds_class, y)
            metric_auprc.update(probs, y)

    return {
        "accuracy": metric_acc.compute().item(),
        "recall": metric_recall.compute().item(),
        "auprc": metric_auprc.compute().item()
    }

if __name__ == "__main__":
    main()
    print("Baseline federated learning process completed successfully.")

