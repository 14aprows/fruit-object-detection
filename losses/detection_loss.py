def compute_loss(loss_dict):
    total_loss = sum(loss for loss in loss_dict.values())
    return total_loss
