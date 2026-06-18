# creates identity matrix
import torch


def identity(batch_size, dim=3, dtype=torch.float32, device='cpu'):
    """
    Creates an identity matrix of shape (batch_size, dim, dim).
    Remember for 2d Points the dim is 3, as affine transformation require the etra dimension for translation
    If batch size is list tuple or torch Size, it will be unpacked
    """
    # if batch size is a unpackable type, unpack it
    id = torch.eye(dim, dtype=dtype, device=device)
    if isinstance(batch_size, (list, tuple, torch.Size)):
        # check for empty batch size
        if len(batch_size) == 0:
            return id
        id_matrix = id.unsqueeze(0).repeat(*batch_size, 1, 1)
    else:
        id_matrix = id.unsqueeze(0).repeat(batch_size, 1, 1)
    return id_matrix
