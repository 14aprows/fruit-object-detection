import torch
import configs.config as cfg
from dataset.dataloader import create_train_dataloader, create_valid_dataloader
from models.faster_rcnn import fasterrcnn_resnet50_fpn
from trainer.trainer import train_model

def seed_everything(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    seed_everything()
    device = torch.device(cfg.DEVICE)

    train_loader = create_train_dataloader()
    valid_loader = create_valid_dataloader()

    model = fasterrcnn_resnet50_fpn(cfg.NUM_CLASSES).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.LR,
        momentum=0.9,
        weight_decay=5e-4
    )

    model = train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        epochs=cfg.NUM_EPOCHS
    )

    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")

if __name__ == "__main__":
    main()