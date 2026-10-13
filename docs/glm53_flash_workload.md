# GLM-5.3-Flash inference workload

从仓库根目录运行（无需模型权重或 AIOB）：

```bash
python -m workload_generator.SimAI_inference_workload_generator glm5.3-flash \
  scripts/inference_configs/glm53_flash_default.json \
  --phase prefill --seq_length 1024 --micro_batch 2 \
  --world_size 8 --tensor_model_parallel_size 2 --expert_model_parallel_size 4 \
  --result_dir results/glm53_flash/
```

将 `--phase prefill` 改为 `--phase decode` 即生成单步 decode workload。
模型名也接受 `GLM-5.3-Flash`；省略配置文件使用随仓库提供的默认配置。
支持直接传入官方嵌套 `text_config` 的 config.json。
脚本入口示例：`bash scripts/inference_workload_with_aiob.sh -m glm5.3-flash -p decode -w 8 -t 2 -e 4 -b 2 -s 1024`。
脚本长选项目前使用下划线，例如 `--model_size`、`--result_dir`；短选项如上。

## 来源和字段映射

2026-10-12 重新核对以下第一手来源：

- [Z.AI 官方权重配置](https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json)
- [官方模型卡](https://huggingface.co/zai-org/GLM-5.3-Flash)：提供 Transformers 实现链接。
- [模型卡链接的 Transformers 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm5_next/modeling_glm5_next.py)：`Glm5NextTextLinearAttention`、`Glm5NextTextIndexer`、`Glm5NextTextAttention`、`Glm5NextTextMoE`、`Glm5NextTextHyperConnection`。

默认 JSON 摘录官方 `text_config` 中与文本 decoder 相关的字段，保持字段名和数值；`linear_attn_config` 保留头数、头维度、门控下界、卷积核大小。
`layer_types` 和 `mlp_layer_types` 逐层决定结构，不按模型名称猜测。
`n_routed_experts` 映射到生成器 `num_experts`（288，不含共享专家），`num_experts_per_tok` 映射到 `moe_router_topk`（8）。CLI 运行参数覆盖配置中的同名运行参数，包括 CLI 默认值。

45 层按官方顺序含 34 个 `attention_kda`、11 个 `attention_dsa`；层 0–2 为 `dense_mlp`，层 3–44 为 `moe_route`、`moe_expert`、`shared_experts`。
KDA 阶段聚合 Q/K/V、短卷积、逐通道 forget gate、beta、KDA 更新、sigmoid gated RMSNorm 和输出投影；DSA 聚合 NoPE MLA 和 k-pool 索引器（top-k 2048、pool 4）。mHC 的 pre/post 在 attention 和 FFN 两侧分别列出，最后包含 hc_head 和 final_norm。

## 建模边界与通信假设

这是文本 decoder 的阶段级通信 workload，计算时间来自已有计时或占位值，不做 FLOPs/显存/参数量估算。阶段 mock 不分配权重。视觉编码器、embedding、LM head、MTP speculative decoding 不在本次 workload 范围；decode 文件代表一个生成步。

- 当前支持 PP=1；避免把完整的 45 层误当作每个 pipeline stage，PP>1 明确报错。
- 激活传输按 BF16（2 字节）建模；权重 FP8 量化不自动代表网络传输 FP8。
- `T = micro_batch * seq_length`（prefill），`T = micro_batch`（decode）。每个 attention/dense MLP 输出在 TP>1 时有一次 `ALLREDUCE`，大小 `2*T*4096` 字节。
- 路由专家沿用本仓库按 TP 分摊 token 的 EP 抽象：dispatch/combine 各为 `2*T*4096*8/TP` 字节；EP>1 时为 `ALLTOALL_EP`。大小是 collective 的逻辑 payload，未加入路由元数据、网络协议开销或链路流量倍率。
- shared expert 采用复制计算假设，mHC 与 norm 为本地计算；不会额外产生 TP collective。TP/EP 为各自通信组大小，均须整除 world_size；不假设 world_size=TP*EP。
- TP=1/EP=1 的相应通信为 `NONE`、大小 0。层号保留在输出名称中，便于逐层核对。

不传 `--aiob_enable` 时，入口仍会尝试读取 `results/aiob_outputs/` 下同配置计时文件；缺失则日志明确说明并使用计算时间 `1`。已有解析器不识别的 KDA/DSA/mHC 阶段也使用 `1`。这些占位时间不代表真实模型性能。此接入不支持生成 AIOB 文件。
