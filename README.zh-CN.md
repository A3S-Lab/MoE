# A3S MoE

<p align="center">
  <strong>Language / 语言:</strong>
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

`a3s-moe` 为 [A3S Power](https://github.com/A3S-Lab/Power) 提供模型自有的
Mixture-of-Experts 推理。Power 仍负责模型无关的调度、权重驻留、
完整性、设备与服务组合。本 crate 拥有架构特定的张量名称、布局、
路由方程、内核、KV cache 语义、分词与生成。

支持的架构包括
[OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924)、
[Qwen3-30B-A3B-Base](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base)，以及
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) 的文本生成路径。
OLMoE 每层 64 个专家、每 token 选 8 个，总参数约 7B，活跃参数约 1B。
Qwen3-MoE 与 Qwen3.6 在独立模型边界上运行，不改变 Power 的模型无关驻留核心。

## 当前状态

M0 数值契约、驻留型 M1 CPU 引擎、有界 M2 专家流式传输、
M3 连续批处理路径、M4 服务路径、M6 加密权重基础、
M7 Qwen3-MoE 路径以及 M8 Qwen3.6 文本路径均已实现并经过测试：

- 兼容 Hugging Face 的 OLMoE 配置解析与严格几何校验。
- 全 softmax top-k 路由，采用 OLMoE 默认的未归一化选中概率。
- 精确的融合 gate/up 顺序、SiLU 激活、down 投影、路由加权与专家归约。
- 覆盖 router logits、选中专家、路由权重与最终 hidden states 的固定 tiny oracle。
- 将精确模型路由转换为 Power 的 `RoutedExpertBatch`，无替换、重排或重新归一化。
- 完整嵌入、RMSNorm、Q/K 归一化、半分割 RoPE、分组查询因果注意力、
  事务性 KV cache、残差解码层、最终 norm 与 LM head。
- 在加载任何张量字节前对已发布 Hugging Face 索引做安全校验，以及刻意全驻留的 F32 正确性加载器。
- GPT-NeoX 分词器加载、受词表约束的编码/解码，以及贪婪 prefill/decode 生成。
- 全量 prefill 与增量 decode 的一致性，以及覆盖 logits 与每层路由的固定完整解码器 oracle。
- 带版本、无损的 F32/BF16 打包专家格式，含严格的头、维度、长度、有限值与 dtype 校验。
- 确定性有界缓冲转换器：原子发布打包检查点，并记录源、稠密与专家集合摘要。
- 完整异步解码器：稠密权重保持驻留，从 Power 获取精确路由专家并集，
  在其他记录加载时计算就绪专家，并恢复规范归约顺序。
- 每个生成请求一个 Power 准入许可、事务性取消、可度量的 cache 边界，
  以及 prefill 与增量贪婪 decode 的驻留对比流式一致性。
- 带独立每会话注意力/KV 状态的不规则连续批、每层一次路由并集专家暂存，
  以及与隔离推理匹配的规范按行输出。
- 基于 Power 执行生命周期的公平贪婪调度器，含有界准入、直接取消令牌回收、
  slot 压缩、精确 KV 状态字节记账，以及仅摘要的 step/生命周期证据。
- 确定性每请求采样：temperature、top-p、top-k、min-p、重复、频率与存在惩罚，
  且不削弱批处理路由并集语义。
- 架构感知的 Power 后端、进程本地模型清单、有界服务队列、稳定增量 UTF-8 解码、
  stop 序列缓冲，以及 HTTP 流被放弃时的取消。
- 独立的 Power 组合服务器，提供 OpenAI chat 与 completion 端点，
  以及记录应用冷/热生成、TTFT、专家读取字节、cache 状态、峰值 RSS
  与隔离驻留 CPU 基线的 JSON 基准。
- 独立认证、可寻址的 AES-256-GCM 稠密与专家集合，通过固定检查点清单
  与明文元数据绑定，并由同一 Power 驻留层级消费。
- `a3s-moe-encrypt` CLI 与类型化加密服务源，使密钥不出现在参数、日志、
  模型清单与解密中间文件中。
- 全驻留 Qwen3-MoE F32 CPU 正确性后端，含其独特的注意力头维度、
  每头 Q/K 归一化、GQA、RoPE、稀疏/稠密层调度、归一化 top-k 策略、
  官方每专家张量或融合导出张量、事务性 KV cache 与贪婪解码。
- 无依赖的完整 Qwen3-MoE 解码器 oracle，覆盖 logits、router logits、
  精确路由、稠密到稀疏层转换、prefill/decode 一致性，以及跨共享 Power 路由边界的滑动注意力。
- 严格的 Qwen3-MoE Hugging Face 分片索引校验：官方 18,867 张量拆分专家布局
  或 531 张量融合导出布局、驻留检查点加载，以及共享的词表约束分词器/流解码器。
- 有界 Qwen3-MoE 转换：逐专家读取官方 gate/up/down 矩阵，
  通过已验证的 Power 张量子范围读取融合三维导出，分块流式处理过大稠密张量，
  并为每个稀疏层专家原子发布一条打包记录。
- Qwen3-MoE 流式解码器：共享驻留注意力、稠密 MLP、归一化与事务性 KV cache 实现，
  同时将全部专家驻留委托给单一 Power 层级。
- 共享连续调度适配器：保留架构特定的路由并集输出类型，
  同时为 OLMoE 与 Qwen3-MoE 复用 Power 的准入、生命周期、取消、采样与 KV 记账。
- Qwen3-MoE Power 后端与服务器组合路径：自动模型族检测、并发请求批处理、
  OpenAI completion/chat 流式传输，以及与 OLMoE 相同的失败关闭请求策略。
- 固定的公开 Qwen3-MoE 验收契约，以及基于固定 BF16 权重的独立 Transformers
  F32 运算 oracle，外加架构感知的 Rust 校验器与性能证据工具。
  已校验的公开 30B-A3B 报告覆盖数值一致性、两个并发 HTTP 请求、
  有界 CPU 推理与原始性能遥测。
- 固定的 Qwen3.6-35B-A3B 契约，针对精确的 26 分片 BF16 检查点，
  含外层多模态索引、693 个文本张量、分词器、修订、字节长度与文件 SHA-256。
  Vision 与 MTP 命名空间为经认证的源输入，但刻意排除在打包检查点的
  `text-generation` 能力之外。
- 精确的 Qwen3.6 文本方程：每四层组中三个 Gated DeltaNet 层后跟一个全注意力层、
  偏移 RMSNorm、每头 Q/K 归一化、部分 RoPE、注意力输出门控、
  事务性混合循环/卷积/KV cache，以及贪婪增量生成。
- 全部 40 个 Qwen3.6 层在 256 个专家上路由归一化 Top-8 选择，
  并与门控共享专家组合。融合专家数组经公共有界子范围打包器转换，
  并由唯一的 Power 权重层级提供服务。
- Qwen3.6 不规则路由并集批处理、类型化后端组合、自动 `qwen3_5_moe` 检测、
  并发 OpenAI completion、SSE、基准证据，以及溯源绑定的公开 F32 运算校验器。
- 在 Power 解析的加速器上，设备原生执行 Qwen3.6 稠密、Gated DeltaNet、注意力、
  共享专家与路由专家，并附带显式无回退的 CUDA 公开检查点一致性与性能报告。

当前 Qwen3.6 实现仅暴露文本生成。它不宣称 vision、video 或 MTP 推理。
HTTP 传输、OpenAI 响应成帧、认证、限流、指标与关闭生命周期仍由 Power 拥有。
精确固定检查点在 `cuda:0` 上通过独立 oracle 一致性验证，无 CPU 回退。
在已校验的 20 逻辑核 Windows 主机、RTX 4090、8 GiB Power 设备 cache、
无主机专家 cache、生成 16 个 token 的条件下，三次热样本平均端到端
`0.263611 tokens/s`、TTFT `3.943 s`。其派生的首 token 后 decode 速率平均
`0.264298 tokens/s`。这是当前无损 F32 执行基线，而非量化或 DSpark 加速结果。
已校验的
[验证](evidence/qwen3.6-35b-a3b-public-validation.json)、
[CUDA 性能](evidence/qwen3.6-35b-a3b-public-cuda-windows.json)、
[CPU 性能](evidence/qwen3.6-35b-a3b-public-cpu-windows.json) 与
[双请求 HTTP](evidence/qwen3.6-35b-a3b-public-http-windows.json) 产物
保留原始数值与精确修订。公开检查点的 Metal 一致性仍待完成。

完整固定的 13.8 GB OLMoE 公开检查点已通过 Transformers 到 Rust 的数值验证、
有界转换、Power 流式生成、驻留 token 一致性比较，以及真实 HTTP
completion/SSE 冒烟测试。已发布的 OLMoE 检查点是基座模型，不声明 chat template，
因此服务器使用显式通用 transcript，除非为兼容微调提供 `--chat-template`。
已校验的原始证据位于 [`evidence/`](evidence/)。

## 架构边界

```text
a3s-moe (model owner)
  model math / tokenizer / generation / speculative drafter adapters
                         |
                         | exact routes + atomic weights + draft/verify contract
                         v
a3s-power (runtime owner)
  admission / batching / residency / speculative scheduling / integrity / TEE / API
```

驻留层级只有一个。模型代码消费 Power 返回的权重，不引入第二个专家 cache。
对 Power 而言，每个打包专家仍是不透明的 `U8` SafeTensor；本 crate 校验并解释
其带版本的头与精确标量载荷。

不变量与交付计划见 [Architecture](docs/architecture.md)；
跨架构 DSpark 边界与验收门见
[Model-neutral speculative decoding](docs/speculative-decoding.md)。

## 开发

```shell
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
python tools/test_generate_public_oracle.py
python tools/test_generate_qwen3_moe_oracle.py
python tools/test_generate_qwen3_moe_full_oracle.py
python tools/test_generate_qwen3_moe_public_oracle.py
python tools/test_qwen3_5_moe_public_contract.py
python tools/test_generate_qwen3_5_moe_public_oracle.py
```

在不下载 13.8 GB 载荷的情况下，校验固定公开检查点的 3,219 个张量头：

```shell
python tools/verify_hf_contract.py
```

从精确固定的 Transformers 检出生成并校验完整公开检查点数值 oracle：

```shell
PYTHONPATH=/src/transformers/src python tools/generate_public_oracle.py \
  /models/OLMoE-1B-7B-0924 tests/fixtures/olmoe_public_oracle.json
cargo run --release --features validation --bin a3s-moe-validate -- \
  /models/OLMoE-1B-7B-0924 tests/fixtures/olmoe_public_oracle.json \
  > olmoe-validation.json
```

生成器拒绝除 `918dbf131d0df5b46e3f6e1d96174d62aa4d16d6` 以外的 Transformers Git 检出，
或 SHA-256 与固定摘要不符的 OLMoE 源文件。它还会在加载模型前校验修订
`6d84c48581ece794365f2b8e9cfb043c68ade9c5` 中每个检查点文件的精确字节长度与 SHA-256。
oracle 绑定该清单，并捕获每个 prompt logit、router logit、选中专家与路由权重。
校验器在溯源、分词器、argmax、路由或容差失败时以非零退出，
并始终为结构有效的数值比较发出带版本的 JSON 报告。

下载 `tools/qwen3_moe_public_contract.py` 中固定的精确
`Qwen/Qwen3-30B-A3B-Base` 修订后，生成独立的公开 Qwen3-MoE oracle，
再与打包的 Power 流式解码器比较：

```shell
PYTHONPATH=/src/transformers/src python \
  tools/generate_qwen3_moe_public_oracle.py \
  /models/Qwen3-30B-A3B-Base qwen3-moe-oracle.json
cargo run --release --features validation --bin a3s-moe-validate -- \
  /models/Qwen3-30B-A3B-Base qwen3-moe-oracle.json \
  --packed-checkpoint /models/Qwen3-30B-A3B-Base-a3s \
  --host-cache-mib 4096 > qwen3-moe-validation.json
```

生成器固定模型修订、全部 16 个分片的字节长度与 SHA-256 摘要、
Transformers Git 修订、其 Qwen3-MoE 源摘要、带每运算 CPU F32 提升的 BF16
参数存储、eager 注意力与专家实现，以及输入 token ID。Rust 校验器在比较
每个捕获的 logit、router logit、路由权重、选中专家与 token argmax 之前，
重新哈希这些文件与打包源绑定。

从 `tools/qwen3_5_moe_public_contract.py` 中固定的精确修订生成并校验
Qwen3.6 文本 oracle：

```shell
PYTHONPATH=/src/transformers/src python \
  tools/generate_qwen3_5_moe_public_oracle.py \
  /models/Qwen3.6-35B-A3B qwen3.6-35b-a3b-oracle.json
cargo run --release --features validation --bin a3s-moe-validate -- \
  /models/Qwen3.6-35B-A3B qwen3.6-35b-a3b-oracle.json \
  --packed-checkpoint /models/Qwen3.6-35B-A3B-a3s \
  --host-cache-mib 4096 > qwen3.6-35b-a3b-validation.json
```

oracle 仅将官方 `model.language_model` 命名空间加载到固定的 Transformers
`Qwen3_5MoeForCausalLM`，拒绝不完整的键映射，禁用可选内核，
将嵌入、线性与深度卷积运算提升到 F32，并捕获 logits 以及每层的归一化 Top-8 路由。
Rust 校验器在检查仅文本打包绑定之前，重新哈希所有源文件，包括被忽略的
vision 与 MTP 分片。

转换已下载的 Hugging Face 检查点，而不缓冲完整层或完整模型：

```shell
cargo run --release --bin a3s-moe-pack -- \
  /models/OLMoE-1B-7B-0924 /models/OLMoE-1B-7B-0924-a3s \
  --experts-per-file 8 --max-buffer-mib 512
```

仅在转换与摘要校验完成后才创建目标。命令拒绝覆盖已有目标，
并输出包含观察到的峰值缓冲字节的 JSON 转换报告。它从已校验的源配置读取
`model_type`，并接受 `olmoe`、`qwen3_moe` 与 `qwen3_5_moe`。每个模型族提供显式的
Power 资源限制：Qwen3-MoE 最多允许 64 GiB 源或打包权重与 8,192 个 SafeTensor 文件，
足以覆盖固定的 61 GB 检查点以及最坏支持的每文件一专家打包，
且不削弱 Power 的全局默认值。其每张量 512M 边界同时覆盖
311,164,928 元素的嵌入/头与 402,653,184 元素的融合 gate/up 导出。
该族的 24 GiB 状态边界覆盖四个完整的 32K F32 KV cache（各 6 GiB）。
其入口驻留配置还允许一个有界的 4 GiB 当前层专家并集，
同时将并发读取限制在 512 MiB 内；这覆盖全部 128 个公开模型专家的无损 F32 形式。
显式库策略在加载期间绝不会被放宽。在物化任何稠密张量之前，
其精确目标 F32 大小会与完整的主机/设备专家 cache 预算一起，
在 Power 的驻留权重限制下被准入。例如：

```shell
cargo run --release --bin a3s-moe-pack -- \
  /models/Qwen3-30B-A3B-Base /models/Qwen3-30B-A3B-Base-a3s \
  --experts-per-file 8 --max-buffer-mib 512
```

Qwen3.6 文本转换使用同一命令。其模型族配置在显式的 80 GiB、16,384 文件、
48 GiB 状态与 4 GiB 暂存边界下，准入固定的 71.9 GB 源、全部 10,240 个可能的
每专家打包文件、四个最大上下文混合 cache，以及完整的 256 专家 F32 层并集：

```shell
cargo run --release --bin a3s-moe-pack -- \
  /models/Qwen3.6-35B-A3B /models/Qwen3.6-35B-A3B-a3s \
  --experts-per-file 8 --max-buffer-mib 512
```

用进程环境提供的密钥加密打包检查点：

```shell
export A3S_MOE_WEIGHT_KEY="$(openssl rand -hex 32)"
cargo run --release --bin a3s-moe-encrypt -- \
  /models/OLMoE-1B-7B-0924-a3s /models/OLMoE-1B-7B-0924-a3s-encrypted \
  --key-env A3S_MOE_WEIGHT_KEY --chunk-mib 1 \
  > olmoe-encryption.json
```

将输出的 `manifestSha256` 存为检查点的带外信任锚。稠密与专家权重被加密；
`config.json`、`manifest.json` 与可选分词器保持明文，但由该受信任清单以
SHA-256 绑定。加密与打开使用有界分块，且从不发布解密中间文件。

通过 Power 的 OpenAI 兼容 API 提供打包检查点服务：

```shell
cargo run --release --features server --bin a3s-moe-server -- \
  /models/OLMoE-1B-7B-0924-a3s \
  --model olmoe-1b-7b --device cpu --host-cache-mib 512 \
  --max-concurrent-requests 4
```

服务器从检查点的有界 `config.json` 检测 `olmoe`、`qwen3_moe` 或 `qwen3_5_moe`，
注入对应的类型化后端，并注册进程本地清单。支持的采样控制为
`temperature`、`top_p`、`top_k`、`min_p`、`seed`、`repeat_penalty`、
`repeat_last_n`、`frequency_penalty` 与 `presence_penalty`。不支持的模态、工具、
结构化输出、跨请求 KV 会话与后端旋钮会在推理前失败。

通过同一二进制为打包的 Qwen3-MoE 检查点提供服务；省略 `--model` 时选择
架构特定的默认标识符 `qwen3-moe`：

```shell
cargo run --release --features server --bin a3s-moe-server -- \
  /models/Qwen3-30B-A3B-Base-a3s --device cpu \
  --host-cache-mib 512 --max-concurrent-requests 4
```

对于 Qwen3.6，省略 `--model` 时选择 `qwen3.6-35b-a3b`：

```shell
cargo run --release --features server --bin a3s-moe-server -- \
  /models/Qwen3.6-35B-A3B-a3s --device cpu \
  --host-cache-mib 4096 --max-concurrent-requests 4
```

加密服务加载当前仅适用于 OLMoE 机密检查点信封，对 Qwen3-MoE 与 Qwen3.6
失败关闭。

通过同时提供其固定信任锚与拥有密钥的环境变量来提供加密形式：

```shell
cargo run --release --features server --bin a3s-moe-server -- \
  /models/OLMoE-1B-7B-0924-a3s-encrypted --model olmoe-1b-7b \
  --encrypted-manifest-sha256 <manifestSha256> \
  --encrypted-key-env A3S_MOE_WEIGHT_KEY
```

生产机密主机应从其经证明的密钥释放机制构造类型化加密源。
基于环境的 CLI 是避免命令行密钥暴露的运维边界；它本身不是远程证明协议。

同一图可在 Power 解析的加速器上运行。精确构建一个平台特性并显式选择
类型化设备：

```shell
cargo run --release --features server,cuda --bin a3s-moe-server -- \
  /models/OLMoE-1B-7B-0924-a3s --device cuda:0 \
  --host-cache-mib 512 --device-cache-mib 4096
```

Metal 在 macOS 上使用 `--features server,metal --device metal:0`。显式的 CUDA 或
Metal 请求在该后端或序号不可用时失败。`auto` 是唯一允许回退到 CPU 的模式，
解析后的设备与回退决策作为无内容的服务/基准证据暴露。

生成原始、可复现的性能证据，并可选择在隔离子进程中与全驻留 CPU 路径比较：

```shell
cargo run --release --features benchmark --bin a3s-moe-bench -- \
  /models/OLMoE-1B-7B-0924-a3s \
  --prompt "Bitcoin is" --max-tokens 32 --warm-samples 5 \
  --checkpoint-label olmoe-1b-7b-bf16 \
  --resident-checkpoint /models/OLMoE-1B-7B-0924 \
  > olmoe-performance.json
```

Qwen3-MoE 使用同一工具与模型族特定的证据模式。其公开一致性门是基于固定
BF16 权重的独立 F32 运算 oracle，因此刻意省略内存密集的驻留 F32 子进程：

```shell
cargo run --release --features benchmark --bin a3s-moe-bench -- \
  /models/Qwen3-30B-A3B-Base-a3s \
  --prompt "Bitcoin is" --max-tokens 8 --warm-samples 3 \
  --host-cache-mib 4096 --checkpoint-label qwen3-30b-a3b-bf16 \
  > qwen3-moe-performance.json
```

Qwen3.6 使用同一证据边界与其自有带版本模式：

```shell
cargo run --release --features benchmark --bin a3s-moe-bench -- \
  /models/Qwen3.6-35B-A3B-a3s \
  --prompt "Bitcoin is" --max-tokens 8 --warm-samples 3 \
  --host-cache-mib 4096 --checkpoint-label qwen3.6-35b-a3b-bf16 \
  > qwen3.6-35b-a3b-performance.json
```

已校验的 CUDA 运行使用显式设备请求与有界设备 cache：

```shell
cargo run --release --features benchmark,cuda --bin a3s-moe-bench -- \
  /models/Qwen3.6-35B-A3B-a3s \
  --prompt "Hello" --max-tokens 16 --warm-samples 3 \
  --device cuda:0 --host-cache-mib 0 --device-cache-mib 8192 \
  --checkpoint-label qwen3.6-35b-a3b-bf16 \
  > qwen3.6-35b-a3b-cuda-performance.json
```

首个样本从空的 Power 专家 cache 开始。热样本仅保留配置的有界 cache。
报告显式将操作系统页 cache 标注为不可控；它不将该条件称为物理冷 I/O。
已校验的 Qwen3.6 CPU 产物报告热均值 `0.194616 tokens/s`；已校验的 CUDA 产物报告
热均值 `0.263611 tokens/s`。测量边界与比较规则见
[Performance Evidence](docs/performance.md)。

CUDA 与 Metal 是互斥的平台特定 Cargo 特性，因此可移植 CI 使用
`--features server,benchmark,validation` 而非 `--all-features`。
CUDA 与 macOS runner 必须分别编译并测试 `cuda` 与 `metal`。

仅在有意更改固定模型契约时重新生成 tiny oracle：

```shell
python tools/generate_tiny_oracle.py
```

生成器将 JSON 打印到 stdout，从不覆盖已校验的 fixture。
在替换 `tests/fixtures/olmoe_tiny_oracle.json` 前请审阅变更。
同一规则适用于 `generate_full_model_oracle.py` 及其完整解码器 fixture。

Qwen3-MoE 稀疏层与完整解码器 fixture 遵循相同的仅审阅工作流：

```shell
python tools/generate_qwen3_moe_oracle.py
python tools/generate_qwen3_moe_full_oracle.py
```

M7 现已为第二模型族提供驻留 CPU 参考推理、严格的官方拆分与融合导出
检查点加载、分词器集成、有界转换、基于 Power 的专家流式传输、
路由并集连续批处理、服务组合，以及固定的公开验收工具。它复用 Power 的
模型无关 `RoutedExpertBatch`、已验证张量范围 I/O、生命周期与唯一驻留层级。
固定的公开模型数值、并发 HTTP 与性能报告已校验于 `evidence/`，
因此 CPU Qwen3-MoE 部署路径在修订 `0146bd6`（Power 修订 `42c6646`）下被接受。

## 许可证

MIT
