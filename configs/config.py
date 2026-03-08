import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4
NUM_EPOCHS = 20
LR = 0.005

NUM_CLASSES = 4

TRAIN_DIR = "Data/train_zip/train"
TEST_DIR = "Data/test_zip/test"