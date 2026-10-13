"""GLM-5.3-Flash text decoder stages (no tensors or timing kernels).

Sources and communication assumptions: docs/glm53_flash_workload.md.
"""

import json
from pathlib import Path

from workload_generator.mocked_model.MockedModel import MockedModel, MockedParamsBase


class GLM53FlashStage(MockedModel):
    def __init__(self, layer_id, name, communication="NONE", **metadata):
        self.layer_id = layer_id
        self.name = name
        self.communication = communication
        self.metadata = metadata


class GLM53FlashTransformerLayer(MockedModel):
    def __init__(self, config, layer_id):
        stages = []

        def add(name, communication="NONE", **metadata):
            stages.append(GLM53FlashStage(layer_id, name, communication, **metadata))

        if config.mhc:
            add("hc_attn_pre", streams=config.hc_mult)
        add("attention_norm")
        kind = config.layer_types[layer_id]
        if kind == "linear_attention":
            # Includes Q/K/V projections, short convolutions, channel-wise forget
            # gate, beta, KDA state update, sigmoid gated RMSNorm and output proj.
            add("attention_kda", "TP", **config.linear_attn_config)
        else:
            # Includes the k-pool indexer and the NoPE MLA attention path.
            add("attention_dsa", "TP", index_topk=config.index_topk,
                index_kpool=config.index_kpool, q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank)
        if config.mhc:
            add("hc_attn_post", streams=config.hc_mult)
            add("hc_ffn_pre", streams=config.hc_mult)
        add("moe_norm")
        if config.mlp_layer_types[layer_id] == "dense":
            add("dense_mlp", "TP", intermediate_size=config.intermediate_size)
        else:
            add("moe_route", "EP", experts=config.n_routed_experts,
                topk=config.num_experts_per_tok)
            add("moe_expert", "EP", intermediate_size=config.moe_intermediate_size)
            # Replicated shared expert: no additional TP collective.
            add("shared_experts", intermediate_size=(
                config.n_shared_experts * config.moe_intermediate_size))
        if config.mhc:
            add("hc_ffn_post", streams=config.hc_mult)
        self.stages = stages


class GLM53FlashModel(MockedModel):
    def __init__(self, config):
        self.layers = [GLM53FlashTransformerLayer(config, i)
                       for i in range(config.num_hidden_layers)]
        self.final_stages = []
        if config.mhc:
            self.final_stages.append(GLM53FlashStage(-1, "hc_head"))
        self.final_stages.append(GLM53FlashStage(-1, "final_norm"))


class GLM53FlashParams(MockedParamsBase):
    def __init__(self, config_file=None, args=None):
        self.world_size = self.tensor_model_parallel_size = 1
        self.expert_model_parallel_size = self.pipeline_model_parallel = 1
        self.micro_batch = self.seq_length = 1
        self.phase = "decode"
        self.moe_enable = True
        self.aiob_enable = False
        self.result_dir = "results/workload/"
        if config_file is None:
            config_file = Path(__file__).resolve().parents[3] / (
                "scripts/inference_configs/glm53_flash_default.json")
        super().__init__("glm5.3-flash", "GLM53Flash", config_file, args)
        self.num_experts = self.n_routed_experts
        self.moe_router_topk = self.num_experts_per_tok
        for field, allowed in (
            ("layer_types", {"linear_attention", "deepseek_sparse_attention"}),
            ("mlp_layer_types", {"dense", "sparse"}),
        ):
            values = getattr(self, field)
            if len(values) != self.num_hidden_layers or not set(values) <= allowed:
                raise ValueError(f"Invalid {field}: expected one supported type per layer")
        for field in ("world_size", "tensor_model_parallel_size",
                      "expert_model_parallel_size", "micro_batch", "seq_length"):
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be positive")
        tp, ep = self.tensor_model_parallel_size, self.expert_model_parallel_size
        if self.world_size % tp or self.world_size % ep:
            raise ValueError("world_size must be divisible by TP and EP")
        if self.num_attention_heads % tp or self.linear_attn_config["num_heads"] % tp:
            raise ValueError("Both attention head counts must be divisible by TP")
        if self.n_routed_experts % ep:
            raise ValueError("Routed expert count must be divisible by EP")
        if self.pipeline_model_parallel != 1:
            raise ValueError("GLM-5.3-Flash workload currently supports PP=1")
        if self.aiob_enable:
            raise ValueError("GLM-5.3-Flash workload requires --aiob_enable to be omitted")

    def load_from_config(self, config_file):
        with open(config_file, encoding="utf-8") as f:
            data = json.load(f)
        # Accept both the official multimodal config and the bundled text config.
        for key, value in data.get("text_config", data).items():
            setattr(self, key, value)
