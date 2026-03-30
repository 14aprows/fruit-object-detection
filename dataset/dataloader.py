from torch.utils.data import DataLoader
from dataset.fruit_dataset import FruitImageDataset
from utils.collate_fn import collate_fn
from dataset.preprocessing import get_train_transform, get_valid_transform
import configs.config as cfg

def create_train_dataloader():
    train_dataset = FruitImageDataset(
        files_dir=cfg.TRAIN_DIR,
        transform=get_train_transform()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    return train_loader

def create_valid_dataloader():
    valid_dataset = FruitImageDataset(
        files_dir=cfg.TEST_DIR,
        transform=get_valid_transform()
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    return valid_loader