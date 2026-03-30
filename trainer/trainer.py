import torch
from losses.detection_loss import compute_loss

def train_one_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]

        targets = [
            {k: v.to(device) for k, v in t.items()}
            for t in targets
        ]

        loss_dict = model(images, targets)
        loss = compute_loss(loss_dict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.train()
    total_loss = 0

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]

            targets = [
                {k: v.to(device) for k, v in t.items()}
                for t in targets
            ]

            loss_dict = model(images, targets)
            loss = compute_loss(loss_dict)

            total_loss += loss.item()

    return total_loss / len(loader)

def train_model(model, optimizer, train_loader, valid_loader, device, epochs):
    for epoch in range(epochs):

        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        valid_loss = evaluate(model, valid_loader, device)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Valid Loss: {valid_loss:.4f}"
        )

    return model