import json

import torch

from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file


def load_config_hf(model_name):
    resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
    return json.load(open(resolved_archive_file))


def load_state_dict_hf(model_name, device=None, dtype=None):
    # If not fp32, then we don't want to load directly onto an accelerator.
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
    try:
        state_dict = torch.load(resolved_archive_file, map_location=mapped_device, weights_only=True)
    except (TypeError, RuntimeError):
        # Older PyTorch versions don't accept ``weights_only`` and very old
        # legacy checkpoints may need the unrestricted loader.
        state_dict = torch.load(resolved_archive_file, map_location=mapped_device)
    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}
    if device is not None:
        state_dict = {k: v.to(device=device) for k, v in state_dict.items()}
    return state_dict
