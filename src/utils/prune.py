import torch


def importance_prune(layer1, layer2, num):
    device = layer1.weight.device
    grad1 = layer1.weight.grad
    grad2 = layer2.weight.grad

    importance1 = layer1.weight * grad1
    importance2 = layer2.weight * grad2

    i_c = importance1.sum(dim=1)
    i_r = importance2.sum(dim=0)

    importance = (i_c + i_r).abs()
    top_k_values, top_k_indices = torch.topk(importance, num, largest=False)
    print(top_k_indices, importance)

    mask1 = torch.ones(layer1.weight.shape[0], device=device)
    mask2 = torch.ones(layer2.weight.shape[1], device=device)

    mask1[top_k_indices] = 0
    mask2[top_k_indices] = 0


    layer1.prune_mask = mask1
    layer2.prune_dim = 1
    layer2.prune_mask = mask2
