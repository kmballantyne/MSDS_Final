import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
PROJECT_ROOT = '/net/scratch/kmballantyne/msds_final'
METADATA_DIR = os.path.join(PROJECT_ROOT, "chexphoto_FL", "metadata")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

# Data/Training Hyperparameters
NUM_CLIENTS = 5
NUM_ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_CLASSES = 2
DEVICE = 'cuda'  # or 'cpu'
IMG_SIZE = (512, 512)  # Image size for input to the model --> could try 512, 512, etc.
OPTIMIZER = "sgd"          # "sgd" or "adam"

# Per-FedAvg settings
# Number of personalization gradient steps per client per round
PERFEDAVG_STEPS = 5
PERFEDAVG_LR = LEARNING_RATE

# Ditto settings
# Proximal regularization strength (lambda / mu)
DITTO_MU = 1e-2
DITTO_LOCAL_EPOCHS = LOCAL_EPOCHS
DITTO_LR = LEARNING_RATE