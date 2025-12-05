import os
import glob
import torch

from model import get_model
# from client import local_train
# from server import average_weights
from utils import load_client_datasets
from perfedavg import per_fedavg_train, personalize
from config import *
import torch

from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import BinaryAUROC 
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryAveragePrecision
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

CKPT_DIR = "checkpoints"
CKPT_PREFIX = "perfedavg"  # so it doesn't clash with fedavg/ditto

def save_checkpoint(global_model, personalized_states, round_idx: int):
    """
    Save a checkpoint after round_idx (1-based for readability).
    Stores both the global model state and the per-client personalized states.
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"{CKPT_PREFIX}_round{round_idx}.pt")
    torch.save(
        {
            "round": round_idx,                  # 1-based round index
            "model_state": global_model.state_dict(),
            "personalized_states": personalized_states,
        },
        ckpt_path,
    )
    print(f"[CKPT] Saved checkpoint at {ckpt_path}")

def load_latest_checkpoint(global_model):
    """
    If any checkpoints exist, load the latest one into global_model and
    return the following:
    - round_idx (0-based) to resume training from
    - personalized_states for all clients
    Otherwise, return 0 to start from scratch with empty personalized states.
    """
    if not os.path.isdir(CKPT_DIR):
        print("[CKPT] No checkpoint directory found, starting from scratch.")
        return 0, [None for _ in range(NUM_CLIENTS)]  # no checkpoints, start from round 0

    pattern = os.path.join(CKPT_DIR, f"{CKPT_PREFIX}_round*.pt")
    ckpts = glob.glob(pattern)
    if not ckpts:
        print("[CKPT] No checkpoints found, starting from scratch.")
        return 0, [None for _ in range(NUM_CLIENTS)]

    def extract_round(path):
        name = os.path.splitext(os.path.basename(path))[0]
        # i.e. "perfedavg_round7" -> "7"
        return int(name.split("round")[-1])

    latest_path = max(ckpts, key=extract_round)
    checkpoint = torch.load(latest_path, map_location=DEVICE)
    round_idx = checkpoint["round"]  # 1-based
    global_model.load_state_dict(checkpoint["global_state"])
    personalized_states = checkpoint.get(
        "personalized_states", [None for _ in range(NUM_CLIENTS)]
    )
    print(f"[CKPT] Loaded checkpoint from {latest_path} (round {round_idx})")

    # training loop uses 0-based, so next round index is round_idx
    # (i.e., if we finished round 3, we start at index 3 -> "Round 4")
    return round_idx, personalized_states

def main():
    # Setup
    print("[DEBUG] Entered PerFedAvg main()")
    global_model = get_model().to(DEVICE)

    # Try to resume from checkpoint if available
    start_round, personalized_states = load_latest_checkpoint(global_model)  # 0-based index
    print(f"[DEBUG] Starting training from round index {start_round} (1-based round {start_round+1})")

    clients_data = load_client_datasets()
    print("[DEBUG] Client datasets loaded (PerFedAvg)")
    
    # writer = SummaryWriter(log_dir="runs/perfedavg_experiment")
    writer = None
    
    
    # To store per-client personalized weights over time
    # personalized_states = [None for _ in range(NUM_CLIENTS)]
    
    # Initialize the global model
    for rnd in range(start_round, NUM_ROUNDS):
        print(f"--- Per-FedAvg Round {rnd + 1}/{NUM_ROUNDS} ---", flush=True)
        
        # 1. One Per-FedAvg meta-update round:
        #    For each client:
        #      - Start from global_model
        #      - Run 'personalization_steps' steps of local personalization
        #    Then average personalized states to update global.
        new_global_state, personalized_states = per_fedavg_train(
            global_model, 
            clients_data,
            personalization_steps=PERFEDAVG_STEPS, # from config
            lr=LEARNING_RATE, # from config
            batch_size=BATCH_SIZE # from config
        )
        global_model.load_state_dict(new_global_state)
        
        # 2. Evaluate global model
        combined = ConcatDataset(clients_data)
        global_metrics = evaluate(global_model, combined)
        
        # global_state = global_model.state_dict()
        # global_updates = []
        
        # 3. Evaluate personalized models per client
        # For collecting new personalized states this round
        personalized_accs = []
        with open("per_client_metrics_perfedavg.csv", "a") as f:
            if rnd == 0 and start_round == 0:
                f.write("round,client,"
                        "global_acc,global_recall,global_auprc,"
                        "personal_acc,personal_recall,personal_auprc\n"
                    )
        
            # 4. Client-side PerFedAvg training
            for cid, ds in enumerate(clients_data):
                # global perf on that client
                g_metrics = evaluate(global_model, ds)
                
                # build personalized model
                pers_model = get_model().to(DEVICE)
                if personalized_states[cid] is not None:
                    pers_model.load_state_dict(personalized_states[cid])
                else:
                    pers_model.load_state_dict(global_model.state_dict())
                    
                # pers_model.load_state_dict(personalized_states[cid])
                
                p_metrics = evaluate(pers_model, ds)
                personalized_accs.append(p_metrics["accuracy"])
                
                # Tensorboard logging
                # writer.add_scalar(f"Client_{cid}/Global_Accuracy", g_metrics["accuracy"], rnd)
                # writer.add_scalar(f"Client_{cid}/Personal_Accuracy", p_metrics["accuracy"], rnd)
                
                # CSV logging with full metrics
                f.write(
                    f"{rnd+1},client_{cid},"
                    f"{g_metrics['accuracy']:.4f},"
                    f"{g_metrics['recall']:.4f},"
                    f"{g_metrics['auprc']:.4f},"
                    f"{p_metrics['accuracy']:.4f},"
                    f"{p_metrics['recall']:.4f},"
                    f"{p_metrics['auprc']:.4f}\n"
                )
                
        avg_personal_acc = sum(personalized_accs) / len(personalized_accs)
        
        # 5. Log global AND avg personalized evals
        # writer.add_scalar("Global/Accuracy", global_metrics["accuracy"], rnd)
        # writer.add_scalar("Global/Recall", global_metrics["recall"], rnd)
        # writer.add_scalar("Global/AUPRC", global_metrics["auprc"], rnd)
        # writer.add_scalar("Personalized/Avg_Accuracy", avg_personal_acc, rnd)
        
        with open("global_metrics_perfedavg.csv", "a") as f:
            if rnd == 0:
                f.write("round,accuracy,recall,auprc,avg_personal_acc\n")
            f.write(
                f"{rnd+1},"
                f"{global_metrics['accuracy']:.4f},"
                f"{global_metrics['recall']:.4f},"
                f"{global_metrics['auprc']:.4f},"
                f"{avg_personal_acc:.4f}\n"  
            )
        
        print(
            f"Round {rnd + 1} — Global Acc: {global_metrics['accuracy']:.4f} | "
            f"Personalized Avg Acc: {avg_personal_acc:.4f}",
            flush=True,
        )

    # Optional: Save model or run final evaluation
    torch.save(global_model.state_dict(), "perfedavg_global_model.pth")
    torch.save({"personalized_states": personalized_states}, "perfedavg_personalized.pth")
    # writer.close()
                    
def evaluate(model, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=32)
    
    metric_acc = BinaryAccuracy().to(DEVICE)
    metric_recall = BinaryRecall().to(DEVICE)
    metric_auprc = BinaryAveragePrecision().to(DEVICE)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]  # Get class 1 probabilities
            preds = torch.argmax(logits, dim=1)

            metric_acc.update(preds, y)
            metric_recall.update(preds, y)
            metric_auprc.update(probs, y)

    return {
        "accuracy": metric_acc.compute().item(),
        "recall": metric_recall.compute().item(),
        "auprc": metric_auprc.compute().item()
    }

if __name__ == "__main__":
    main()
    print("Personalized federated learning process completed successfully.")


