import torch
import torch.nn as nn
import torch.nn.functional as F


def entropy_loss(x, ignore_mask):
    b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    b = b.sum(dim=2)
    b[ignore_mask] = 0 # Mask after sum to avoid memory issue.
    b = -1.0 * b.sum(dim=0).mean() # Sum along words and mean along batch
    return b


def global_reconstruction_loss(x, x_recon, keep_mask):
    x = x.mean(dim=1)

    caption_len = keep_mask.sum(dim=0)
    caption_len = caption_len.unsqueeze(1).expand(caption_len.size(0), x_recon.size(2))
    caption_len = caption_len.type(torch.cuda.FloatTensor)
    keep_mask = keep_mask.transpose(0, 1).unsqueeze(2).expand_as(x_recon).type(torch.cuda.FloatTensor)
    x_recon = keep_mask * x_recon
    x_recon = x_recon.sum(dim=1) / caption_len

    l2_loss = F.mse_loss(x, x_recon)
    return l2_loss


def local_reconstruction_loss(x, x_recon):
    l2_loss = F.mse_loss(x, x_recon)
    return l2_loss

