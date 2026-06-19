# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
import os
import json
from functools import partial
from typing import Optional
from collections import namedtuple

import torch
import torch.nn as nn

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except (ImportError, ModuleNotFoundError):
    # Fall back to pure-PyTorch implementations (CPU / MPS) when Triton or
    # the fused CUDA kernels are not available on the host platform.
    from mamba_ssm.ops.cpu_fallbacks import RMSNorm, layer_norm_fn, rms_norm_fn


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        self.embedding_proj = nn.Linear(d_model, d_model, **factory_kwargs)
        self.injection_proj = nn.Linear(d_model, d_model, **factory_kwargs)
        # Initialise the ESI projections so that the search-and-inject step
        # is a *small* perturbation of the residual stream at init time. This
        # makes the model start close to vanilla Mamba behaviour and gives
        # the fine-tuning of the ESI extras (see ``evals/train_esi.py``) a
        # well-conditioned starting point. The default ``nn.Linear`` init is
        # Kaiming-uniform; we shrink the injection magnitude by 10x and the
        # query-side projections by 10x as well.
        with torch.no_grad():
            self.injection_proj.weight.mul_(0.1)
            if self.injection_proj.bias is not None:
                self.injection_proj.bias.zero_()
            self.embedding_proj.weight.mul_(0.1)
            if self.embedding_proj.bias is not None:
                self.embedding_proj.bias.zero_()
        # Number of input tokens to retrieve for embedding-search injection.
        self.esi_top_k = 5
        # When False, the ESI search-and-inject step is skipped entirely, which
        # makes the backbone behave like vanilla Mamba. Useful for ablation /
        # comparison studies and for running eval harnesses that don't pass a
        # question.
        self.esi_enabled = True

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, question_embedding, inference_params=None, return_top_indices=False):
        hidden_states = self.embedding(input_ids)
        top_indices = None
        if self.esi_enabled and question_embedding is not None:
            # Embedding search and injection: score each context token against
            # the (mean-pooled) question embedding, keep the top-k tokens, and
            # add their projected hidden states back into the residual stream.
            #
            # Soft-gated variant: the top-k *similarity scores* are kept and
            # passed through a softmax to produce a (B, top_k) weight vector
            # over the gathered tokens. The weights are differentiable, so
            # gradients can flow back through ``embedding_proj`` and
            # ``question_embedding``. Indices themselves are still discrete.
            input_embeddings = self.embedding_proj(hidden_states)
            similarity_scores = torch.einsum('bd,btd->bt', question_embedding, input_embeddings)
            top_k = min(similarity_scores.size(-1), self.esi_top_k)
            top_scores, top_indices = torch.topk(similarity_scores, top_k, dim=-1)
            weights = top_scores.softmax(dim=-1)  # (B, top_k), differentiable
            relevant_hidden_states = hidden_states.gather(
                1, top_indices.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
            )
            injection_vectors = self.injection_proj(relevant_hidden_states) * weights.unsqueeze(-1)
            aggregated_injection = injection_vectors.sum(1).unsqueeze(1)
            hidden_states = hidden_states + aggregated_injection

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        if return_top_indices:
            return hidden_states, top_indices
        return hidden_states


class MambaLMHeadModel(nn.Module, GenerationMixin):

    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        self.question_embedding = nn.Linear(config.d_model, config.d_model, **factory_kwargs)
        # Same small-perturbation init as ``embedding_proj`` / ``injection_proj``
        # so a freshly constructed MambaESI behaves close to vanilla Mamba.
        with torch.no_grad():
            self.question_embedding.weight.mul_(0.1)
            if self.question_embedding.bias is not None:
                self.question_embedding.bias.zero_()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(
        self,
        input_ids,
        question_ids=None,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        return_top_indices=False,
    ):
        """Forward pass.

        Args:
            input_ids: ``(batch, seqlen)`` context token ids.
            question_ids: optional ``(batch, qlen)`` question token ids. When
                omitted (or when ``backbone.esi_enabled`` is ``False``), the
                ESI search-and-inject step is skipped and the model behaves
                like vanilla Mamba. This keeps the API drop-in compatible with
                upstream lm-eval-harness call sites.
            position_ids: ignored; kept for transformer-style API compatibility.
            num_last_tokens: if > 0, only return logits for the trailing tokens.
            return_top_indices: also return the indices of the top-k retrieved
                context tokens. Useful for inspecting what ESI selected.
        """
        if question_ids is not None and self.backbone.esi_enabled:
            question_embedding = self.question_embedding(
                self.backbone.embedding(question_ids)
            ).mean(1)
        else:
            question_embedding = None
        backbone_out = self.backbone(
            input_ids,
            question_embedding,
            inference_params=inference_params,
            return_top_indices=return_top_indices,
        )
        if return_top_indices:
            hidden_states, top_indices = backbone_out
        else:
            hidden_states, top_indices = backbone_out, None
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        if return_top_indices:
            CausalLMOutput = namedtuple("CausalLMOutput", ["logits", "top_indices"])
            return CausalLMOutput(logits=lm_logits, top_indices=top_indices)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, strict=False, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        # MambaConfig gained a ``tie_embeddings`` field that older HF configs
        # don't have; filter unknown keys so loading legacy checkpoints works.
        allowed = set(MambaConfig.__dataclass_fields__.keys())
        config_data = {k: v for k, v in config_data.items() if k in allowed}
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)
        # The ESI extras (question_embedding, backbone.embedding_proj,
        # backbone.injection_proj) are absent from upstream Mamba checkpoints;
        # use non-strict loading and report what was added so behaviour is
        # transparent.
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if missing:
            print(f"[MambaESI] randomly initialised (missing from checkpoint): {missing}")
        if unexpected:
            print(f"[MambaESI] unexpected keys in checkpoint: {unexpected}")
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f)
