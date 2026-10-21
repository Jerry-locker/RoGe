import torch


def decomposite(tensor_A, tensor_B, rank):
    dtype, device = tensor_A.dtype, tensor_A.device
    weight = tensor_B @ tensor_A
    U, S, V = torch.pca_lowrank(weight.float(), q=rank)
    tensor_A = (V.T).to(dtype=dtype, device=device).contiguous()
    tensor_B = (U @ torch.diag(S)).to(dtype=dtype, device=device).contiguous()
    return tensor_A, tensor_B


def reset_lora_rank(lora, rank):
    lora_merged = {}
    keys = [i for i in lora.keys() if ".lora_A." in i]
    for key in keys:
        tensor_A = lora[key]
        tensor_B = lora[key.replace(".lora_A.", ".lora_B.")]
        tensor_A, tensor_B = decomposite(tensor_A, tensor_B, rank)
        lora_merged[key] = tensor_A
        lora_merged[key.replace(".lora_A.", ".lora_B.")] = tensor_B
    return lora_merged


def infer_lora_rank(state_dict):
    """Infer LoRA rank from ``lora_A`` tensors (first dimension)."""
    ranks = set()
    for key, value in state_dict.items():
        if ".lora_A." in key and isinstance(value, torch.Tensor):
            ranks.add(value.shape[0])
    if not ranks:
        return None
    if len(ranks) > 1:
        raise ValueError(f"Inconsistent LoRA ranks in checkpoint: {sorted(ranks)}")
    return next(iter(ranks))


def adapt_lora_rank(state_dict, target_rank):
    """SVD-adapt LoRA weights when checkpoint rank differs from ``target_rank``."""
    ckpt_rank = infer_lora_rank(state_dict)
    if ckpt_rank is None:
        return state_dict
    if ckpt_rank == target_rank:
        return state_dict
    print(
        f"LoRA rank mismatch: checkpoint rank={ckpt_rank}, target rank={target_rank}. "
        f"Applying SVD rank adaptation."
    )
    return reset_lora_rank(state_dict, target_rank)