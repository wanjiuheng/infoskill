"""
utils/embedding.py

Shared helper for converting text → dense vector using the LLM's own
token-embedding layer (frozen) + mean pooling.

Decision: use the backbone's get_input_embeddings() with mean pooling over
tokens. No separate embedding model needed; hidden_size = 3584 for
Qwen2.5-7B-Instruct.
"""

import torch
import torch.nn.functional as F
from typing import List, Union


@torch.no_grad()
def get_text_embedding(
    text: Union[str, List[str]],
    model,
    tokenizer,
    device: torch.device,
    max_length: int = 256,
) -> torch.Tensor:
    """
    Encode text into a fixed-size dense vector via the LLM's embedding layer.

    Args:
        text:       A single string or list of strings.
        model:      The LLM (Qwen2.5-VL or any HuggingFace model with
                    get_input_embeddings()).  Must already be on `device`.
        tokenizer:  Matching HuggingFace tokenizer.
        device:     Target device.
        max_length: Tokenisation max-length (truncate silently).

    Returns:
        Tensor of shape [1, hidden_size] for a single string,
        or [N, hidden_size] for a list of N strings.
        The embedding layer weights are NOT updated (no_grad + detach).
    """
    single = isinstance(text, str)
    if single:
        text = [text]

    enc = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    # Token embeddings only — no transformer forward pass
    embed_layer = model.get_input_embeddings()
    token_embeds = embed_layer(enc.input_ids)          # [N, seq_len, hidden]

    # Mean-pool over non-padding tokens
    mask = enc.attention_mask.unsqueeze(-1).float()    # [N, seq_len, 1]
    summed = (token_embeds * mask).sum(dim=1)          # [N, hidden]
    counts = mask.sum(dim=1).clamp(min=1e-9)           # [N, 1]
    mean_emb = summed / counts                         # [N, hidden]

    return mean_emb[0] if single else mean_emb


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between every row of `a` and every row of `b`.

    Args:
        a: [M, D]
        b: [N, D]

    Returns:
        [M, N] similarity matrix.
    """
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    return a_norm @ b_norm.T
