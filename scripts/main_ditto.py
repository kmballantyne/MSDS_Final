import numpy as np
from model import get_model
from utils import load_client_datasets
from ditto import ditto_local_train
from server import average_weights
from config import *

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryRecall,
    BinaryAveragePrecision,
)

def evaluate(model, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=32)  # or EVAL_BATCH_SIZE

    metric_acc = BinaryAccuracy().to(DEVICE)
    metric_recall = BinaryRecall().to(DEVICE)
    metric_auprc = BinaryAveragePrecision().to(DEVICE)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            metric_acc.update(preds, y)
            metric_recall.update(preds, y)
            metric_auprc.update(probs, y)

    return {
        "accuracy": float(metric_acc.compute()),
        "recall": float(metric_recall.compute()),
        "auprc": float(metric_auprc.compute()),
    }

def main():
    global_model = get_model().to(DEVICE)
    clients_data = load_client_datasets()

    # writer = SummaryWriter(log_dir="runs/ditto_experiment")

    # Personal models per client: start as None → will default to global on first round
    personal_states = [None for _ in range(NUM_CLIENTS)]

    for rnd in range(NUM_ROUNDS):
        print(f"----- Ditto Round {rnd + 1}/{NUM_ROUNDS} -----")

        global_state = global_model.state_dict()
        client_global_states = []
        new_personal_states = []

        # ---------- Client-side Ditto updates ----------
        for cid in range(NUM_CLIENTS):
            print(f" --- Client {cid + 1}/{NUM_CLIENTS} ---")

            # Global copy for this client
            client_global = get_model().to(DEVICE)
            client_global.load_state_dict(global_state)

            # Personal model: use previous personalized state if available, else start from global
            client_personal = get_model().to(DEVICE)
            if personal_states[cid] is not None:
                client_personal.load_state_dict(personal_states[cid])
            else:
                client_personal.load_state_dict(global_state)

            g_state_new, p_state_new = ditto_local_train(
                global_model=client_global,
                personal_model=client_personal,
                dataset=clients_data[cid],
                mu=DITTO_MU,
                local_epochs=DITTO_LOCAL_EPOCHS,
                lr=DITTO_LR,
                batch_size=BATCH_SIZE,
            )

            client_global_states.append(g_state_new)
            new_personal_states.append(p_state_new)

        # ---------- Aggregate global weights ----------
        print("--- Aggregating global weights (Ditto) ---")
        avg_global_state = average_weights(client_global_states)
        global_model.load_state_dict(avg_global_state)
        personal_states = new_personal_states

        # ---------- Evaluate global model ----------
        combined_dataset = ConcatDataset(clients_data)
        global_metrics = evaluate(global_model, combined_dataset)

        # writer.add_scalar("Global/Accuracy", global_metrics["accuracy"], rnd)
        # writer.add_scalar("Global/Recall", global_metrics["recall"], rnd)
        # writer.add_scalar("Global/AUPRC", global_metrics["auprc"], rnd)

        print(
            f"[Round {rnd + 1}] Global Acc: {global_metrics['accuracy']:.4f}, "
            f"Recall: {global_metrics['recall']:.4f}, "
            f"AUPRC: {global_metrics['auprc']:.4f}"
        )

        # ---------- Evaluate per-client: global vs personalized ----------
        per_client_global_acc = []
        per_client_personal_acc = []

        with open("per_client_metrics_ditto.csv", "a") as f:
            if rnd == 0:
                f.write(
                    "round,client,"
                    "global_acc,global_recall,global_auprc,"
                    "personal_acc,personal_recall,personal_auprc\n"
                )

            for cid, ds in enumerate(clients_data):
                # Global model on this client's data
                g_metrics = evaluate(global_model, ds)
                per_client_global_acc.append(g_metrics["accuracy"])

                # Personalized model on this client's data
                pers_model = get_model().to(DEVICE)
                pers_model.load_state_dict(personal_states[cid])
                p_metrics = evaluate(pers_model, ds)
                per_client_personal_acc.append(p_metrics["accuracy"])

                # writer.add_scalar(
                #     f"Client_{cid}/Global_Accuracy", g_metrics["accuracy"], rnd
                # )
                # writer.add_scalar(
                #     f"Client_{cid}/Personal_Accuracy", p_metrics["accuracy"], rnd
                # )

                f.write(
                    f"{rnd+1},client_{cid},"
                    f"{g_metrics['accuracy']:.4f},"
                    f"{g_metrics['recall']:.4f},"
                    f"{g_metrics['auprc']:.4f},"
                    f"{p_metrics['accuracy']:.4f},"
                    f"{p_metrics['recall']:.4f},"
                    f"{p_metrics['auprc']:.4f}\n"
                )

        # Aggregate fairness-ish stats: mean/std of client accuracies

        global_mean = float(np.mean(per_client_global_acc))
        global_std = float(np.std(per_client_global_acc))
        personal_mean = float(np.mean(per_client_personal_acc))
        personal_std = float(np.std(per_client_personal_acc))

        # writer.add_scalar("Fairness/Global_Acc_Mean", global_mean, rnd)
        # writer.add_scalar("Fairness/Global_Acc_Std", global_std, rnd)
        # writer.add_scalar("Fairness/Personal_Acc_Mean", personal_mean, rnd)
        # writer.add_scalar("Fairness/Personal_Acc_Std", personal_std, rnd)

        with open("global_metrics_ditto.csv", "a") as f:
            if rnd == 0:
                f.write(
                    "round,accuracy,recall,auprc,"
                    "global_acc_mean,global_acc_std,"
                    "personal_acc_mean,personal_acc_std\n"
                )
            f.write(
                f"{rnd+1},"
                f"{global_metrics['accuracy']:.4f},"
                f"{global_metrics['recall']:.4f},"
                f"{global_metrics['auprc']:.4f},"
                f"{global_mean:.4f},{global_std:.4f},"
                f"{personal_mean:.4f},{personal_std:.4f}\n"
            )

    torch.save(global_model.state_dict(), "ditto_global_model.pth")
    torch.save({"personal_states": personal_states}, "ditto_personal_models.pth")
    # writer.close()

if __name__ == "__main__":
    main()
    print("Ditto training completed.")
