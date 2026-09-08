# A3S MoE

<p>
  <strong>Language / 语言:</strong>
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>


`a3s-moe` 提供模型拥有的专家混合推理
[A3S Power](https://github.com/A3S-Lab/Power)。权力仍然负责
模型中立的调度、权重驻留、完整性、设备和服务
组成。这个箱子拥有特定于架构的张量名称、布局、
路由方程、内核、KV 缓存语义、标记化和生成。

支持的架构是
[OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924),
[Qwen3-30B-A3B-Base](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base)，以及
文本生成路径
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)。 OLMoE 有 64
每层专家，每个代币选择 8 个专家，总参数 7B 个，以及
大约 1B 个活动参数。 Qwen3-MoE与Qwen3.6独立运行
模型边界而不改变 Power 的模型中立驻留核心。

## 目前状态

M0数值合约，常驻M1 CPU引擎，有界M2专家流，
M3连续批处理路径、M4服务路径、M6加密权重基础、
M7 Qwen3-MoE路径、M8 Qwen3.6文本路径已实现并测试：

- Hugging Face 兼容 OLMoE 配置解析和严格的几何形状
  验证。
- 带有 OLMoE 默认非标准化选择的 Full-softmax top-k 路由
  概率。
- 精确融合门/上排序、SiLU 激活、下投影、路线
  加权和专家减少。
- 一个固定的微型预言机，涵盖路由器逻辑、选定的专家、路由权重、
  和最终的隐藏状态。
- 将精确模型路线转换为 Power 的`RoutedExpertBatch`，无需
  替换、重新排序或重新规范化。
- 完整嵌入、RMSNorm、Q/K 归一化、分半 RoPE、分组
  查询因果注意力、事务性 KV 缓存、剩余解码器层、
  最终标准，以及LM头。
- 在任何张量字节之前安全验证已发布的 Hugging Face 索引
  已加载，加上故意完全驻留的 F32 正确性加载程序。
- GPT-NeoX 分词器加载、词汇绑定编码/解码和贪婪
  预填充/解码生成。
- 完全预填充与增量解码奇偶校验和固定完整解码器
  oracle 涵盖 logits 和每一层的路线。
- 版本化无损 F32/BF16 打包专家格式，具有严格的标头，
  尺寸、长度、有限值和数据类型验证。
- 发布打包检查点的确定性有界缓冲区转换器
  原子地记录源、密集和专家集合摘要。
- 一个完整的异步解码器，保持密集权重驻留，获得
  来自 Power 的精确路由专家联盟，计算就绪专家，而其他
  记录负载，并恢复规范的归约顺序。
- 每代请求一个电力准入许可证，交易取消，
  测量的缓存边界以及预填充和流式驻留奇偶校验
  增量贪婪解码。
- 具有独立的每会话注意力/KV 状态的参差不齐的连续批次，
  一条路线 - 每层联合专家分段操作，每行规范
  输出匹配孤立的推理。
- 一个基于 Power 执行生命周期的公平贪婪调度程序，包括
  有界入场、直接取消令牌收获、时隙压缩、
  精确的 KV 状态字节计算，以及仅摘要的步骤/生命周期证据。
- 确定性每个请求采样，包括温度、top-p、top-k、min-p、
  不削弱批量路由的重复、频率和存在惩罚
  联合语义。
- 架构感知的 Power 后端、流程本地模型清单、有界
  服务队列、稳定增量UTF-8解码、停止序列缓冲、
  以及当 HTTP 流被放弃时取消。
- 用于 OpenAI 聊天和完成端点的独立 Power-compose 服务器，
  加上记录应用程序冷代和热代的 JSON 基准，
  TTFT、专家字节读取、缓存状态、峰值 RSS 和独立的常驻 CPU
  基线。
- 独立认证、可搜索的 AES-256-GCM 密集且专家
集合，通过固定检查点与纯文本元数据绑定在一起
  通过相同的 Power 驻留层次结构来体现和使用。
- `a3s-moe-encrypt` CLI 和类型化加密服务源，可防止密钥泄露
  参数、日志、模型清单和解密的中间文件。
- 完全驻留的 Qwen3-MoE F32 CPU 正确性后端，具有独特的功能
  注意力头维度、每头 Q/K 归一化、GQA、RoPE、
  稀疏/密集层调度、标准化 top-k 策略、官方每位专家
  张量或融合导出器张量、事务性 KV 缓存和贪婪
  解码。
- 一个无依赖性的完整 Qwen3-MoE 解码器预言机，涵盖 logits、路由器
  logits、精确路径、密集到稀疏层转换、预填充/解码
  奇偶校验，以及在共享 Power 路由边界上滑动注意力。
- 对官方的严格 Qwen3-MoE Hugging Face 分片索引验证
  18,867 张量分割专家布局或 531 张量融合导出器布局，
  常驻检查点加载和共享词汇限制
  分词器/流解码器。
- 有界 Qwen3-MoE 转换，读取官方门/上/下矩阵一
  一次专家，通过验证的功率张量读取融合的 3D 导出器
  子范围，以块的形式传输超大的密集张量，并发布一个
  每个稀疏层专家的原子打包记录。
- 共享居民关注的Qwen3-MoE流解码器，密集的MLP，
  标准化和委托时的事务性 KV 缓存实现
  所有专家都隶属于一个权力层级。
- 保留特定于架构的共享连续调度适配器
  路由联合输出类型，同时重用电源准入、生命周期、
  OLMoE 和 Qwen3-MoE 的取消、采样和 KV 计算。
- 具有自动模型的 Qwen3-MoE Power 后端和服务器组成路径
  家庭检测、并发请求批处理、OpenAI 完成/聊天
  流，以及与 OLMoE 相同的失败关闭请求策略。
- 固定的公开 Qwen3-MoE 验收合同和独立的 Transformers
  F32-对固定 BF16 权重的操作预言机，加上
  架构感知 Rust 验证器和性能证据工具。的
  检查公共 30B-A3B 报告涵盖数字奇偶性、两个并发 HTTP
  请求、有限 CPU 推理和原始性能遥测。
- 固定的 Qwen3.6-35B-A3B 合约，用于精确的 26 分片 BF16 检查点，
  包括其外部多模态索引、693 个文本张量、分词器、修订版、
  字节长度和文件 SHA-256 值。 Vision 和 MTP 命名空间是
  经过身份验证的源输入，但故意从打包中排除
  检查点的`text-generation`能力。
- 精确的 Qwen3.6 文本方程：三个门控 DeltaNet 层，后跟一层
  每四层组的全注意力层，偏移 RMSNorm，每头 Q/K
标准化、部分 RoPE、注意力输出门、事务混合
  循环/卷积/KV 缓存，以及贪婪增量生成。
- 所有 40 个 Qwen3.6 层均经过 256 位专家的标准化 Top-8 选择
  并将它们与门控共享专家结合起来。融合专家数组是
  通过公共有界子范围打包程序进行转换并由唯一的服务
  权力权重等级。
- Qwen3.6 参差不齐的路线联合批处理，类型化后端组合，自动
  `qwen3_5_moe` 检测、并发 OpenAI 完成、SSE、基准测试
  证据，以及受来源约束的公共 F32 操作验证器。
- 设备原生 Qwen3.6 密集、门控 DeltaNet、注意力、共享专家和
  Power-resolved 加速器上的路由专家执行，具有明确的
  无后备 CUDA 公共检查点奇偶校验和性能报告。

Qwen3.6 实现当前仅公开文本生成。它不
宣传视觉、视频或 MTP 推理。 HTTP 传输、OpenAI 响应
框架、身份验证、速率限制、指标和关闭生命周期仍然存在
归权力所有。确切的固定检查点通过了独立预言机奇偶校验
在 `cuda:0` 上，没有 CPU 回退。在检查的 20 逻辑核心 Windows 主机上
配备 RTX 4090、8 GiB Power 设备缓存、无主机专家缓存和 16
生成的令牌，三个温暖样本平均`0.263611 tokens/s`端到端和
`3.943 s` TTFT。他们得出的后第一个令牌解码率平均值
`0.264298 tokens/s`。这是当前无损 F32 执行基线，而不是
量化或 DSpark 加速的结果。已检查的
[validation](evidence/qwen3.6-35b-a3b-public-validation.json),
[CUDA performance](evidence/qwen3.6-35b-a3b-public-cuda-windows.json),
[CPU performance](evidence/qwen3.6-35b-a3b-public-cpu-windows.json)，以及
[two-request HTTP](evidence/qwen3.6-35b-a3b-public-http-windows.json)神器
保留原始值和精确修订。公共检查点金属平价
仍然悬而未决。

完整的固定 13.8 GB OLMoE 公共
检查点已通过 Transformers-to-Rust 数值验证，有界
转换、电力流发电、居民代币平价比较，以及
真正的 HTTP 完成/SSE 冒烟测试。已发布的 OLMoE 检查点是基础
模型并且没有声明聊天模板，因此服务器使用显式
通用转录本，除非为兼容的版本提供了`--chat-template`
微调。检查过的原始证据属于[`evidence/`](evidence/)。

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

有一个居住等级。模型代码消耗由返回的权重
电源并没有引入第二个专家缓存。每个打包的专家都留下来
一个不透明的 `U8` SafeTensor to Power；这个箱子验证并解释了它的
版本化标头和精确标量有效负载。

请参阅[Architecture](docs/architecture.md)了解不变量和交付计划，
和 [Model-neutral speculative decoding](docs/speculative-decoding.md) 为
跨架构 DSpark 边界和接受门。

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

无需下载即可验证固定公共检查点的 3,219 个张量标头
13.8 GB 有效负载：

```shell
python tools/verify_hf_contract.py
```

从准确的数据中生成并验证完整的公共检查点数字预言
固定变形金刚结帐：

```shell
PYTHONPATH=/src/transformers/src python tools/generate_public_oracle.py \
  /models/OLMoE-1B-7B-0924 tests/fixtures/olmoe_public_oracle.json
cargo run --release --features validation --bin a3s-moe-validate -- \
  /models/OLMoE-1B-7B-0924 tests/fixtures/olmoe_public_oracle.json \
  > olmoe-validation.json
```

生成器拒绝 Transformers Git 签出，除了
`918dbf131d0df5b46e3f6e1d96174d62aa4d16d6`，或 OLMoE 源文件，其
SHA-256 与固定摘要不同。它还验证确切的字节长度
以及修订版中每个检查点文件的 SHA-256
加载模型之前的`6d84c48581ece794365f2b8e9cfb043c68ade9c5`。神谕
绑定该清单并捕获每个提示 logit、路由器 logit、选定的
专家和路线权重。验证器在出处、分词器上以非零值退出，
argmax、路由或容差失败，并始终发出版本化 JSON 报告
进行结构上有效的数值比较。

下载准确的Qwen3-MoE预言机后生成独立的公共Qwen3-MoE预言机
`Qwen/Qwen3-30B-A3B-Base` 修订已固定
`tools/qwen3_moe_public_contract.py`，然后与打包后的进行比较
动力流解码器：

```shell
PYTHONPATH=/src/transformers/src python \
  tools/generate_qwen3_moe_public_oracle.py \
  /models/Qwen3-30B-A3B-Base qwen3-moe-oracle.json
cargo run --release --features validation --bin a3s-moe-validate -- \
  /models/Qwen3-30B-A3B-Base qwen3-moe-oracle.json \
  --packed-checkpoint /models/Qwen3-30B-A3B-Base-a3s \
  --host-cache-mib 4096 > qwen3-moe-validation.json
```

生成器固定模型修订版、所有 16 个分片字节长度和 SHA-256
摘要，Transformers Git 修订版，其 Qwen3-MoE 源摘要，BF16
参数存储与每操作CPU F32促销，热切关注和
专家实现和输入令牌 ID。 Rust 验证器
在比较每个文件之前重新散列这些文件和打包的源绑定
捕获的 logit、路由器 logit、路由权重、所选专家和令牌 argmax。

根据固定的确切修订版生成并验证 Qwen3.6 文本预言机
`tools/qwen3_5_moe_public_contract.py`：

```shell
PYTHONPATH=/src/transformers/src python \
  tools/generate_qwen3_5_moe_public_oracle.py \
  /models/Qwen3.6-35B-A3B qwen3.6-35b-a3b-oracle.json
cargo run --release --features validation --bin a3s-moe-validate -- \
  /models/Qwen3.6-35B-A3B qwen3.6-35b-a3b-oracle.json \
  --packed-checkpoint /models/Qwen3.6-35B-A3B-a3s \
  --host-cache-mib 4096 > qwen3.6-35b-a3b-validation.json
```

预言机仅将官方的`model.language_model`命名空间加载到
固定变形金刚`Qwen3_5MoeForCausalLM`，拒绝不完整的键映射，
禁用可选内核，促进嵌入、线性和深度卷积
F32 的操作，并捕获 logits 以及来自每个的标准化 Top-8 路由
层。 Rust 验证器重新散列所有源文件，包括被忽略的文件
在检查纯文本打包绑定之前，先检查视觉和 MTP 分片。

转换下载的 Hugging Face 检查点而不缓冲完整的
层或模型：

```shell
cargo run --release --bin a3s-moe-pack -- \
  /models/OLMoE-1B-7B-0924 /models/OLMoE-1B-7B-0924-a3s \
  --experts-per-file 8 --max-buffer-mib 512
```

仅在转换和摘要验证之后才创建目标
完成。该命令拒绝覆盖现有目标并发出
JSON 转换报告包含观察到的峰值缓冲字节。上面写着
来自已验证的源配置的`model_type`并接受`olmoe`，
`qwen3_moe`和`qwen3_5_moe`。每个家庭都提供明确的电力资源
限制：Qwen3-MoE 允许最多 64 GiB 的源或打包重量和 8,192
安全张量文件，
其中涵盖固定的 61 GB 检查点和最差支持的检查点
每个文件一个专家打包，而不会削弱 Power 的全局默认设置。其
每个张量 512M 范围涵盖 311,164,928 元素嵌入/头和
402,653,184 元件融合栅极/向上导出。该系列的 24 GiB 状态绑定
涵盖四个完整的 32K F32 KV 缓存（每个 6 GiB）。它的入口点驻留
配置文件还允许 1 个有限的 4 GiB
当前层专家联合，同时将并发读取保持在 512 MiB 以内；这个
涵盖无损 F32 形式的所有 128 个公共模型专家。显式库
加载期间策略永远不会扩大。在任何稠密张量之前
实现后，其确切的目标 F32 尺寸与完整的尺寸一起被承认
Power 驻留权重限制下的主机/设备专家缓存预算。例如：

```shell
cargo run --release --bin a3s-moe-pack -- \
  /models/Qwen3-30B-A3B-Base /models/Qwen3-30B-A3B-Base-a3s \
  --experts-per-file 8 --max-buffer-mib 512
```

Qwen3.6文本转换使用相同的命令。其家族概况承认
固定的 71.9 GB 源，所有 10,240 个可能的单专家打包文件，四个
最大上下文混合缓存，以及完整的 256 专家 F32 层联合
显式 80 GiB、16,384 个文件、48 GiB 状态和 4 GiB 暂存边界：

```shell
cargo run --release --bin a3s-moe-pack -- \
  /models/Qwen3.6-35B-A3B /models/Qwen3.6-35B-A3B-a3s \
  --experts-per-file 8 --max-buffer-mib 512
```

使用进程环境提供的密钥加密打包检查点：

```shell
export A3S_MOE_WEIGHT_KEY="$(openssl rand -hex 32)"
cargo run --release --bin a3s-moe-encrypt -- \
  /models/OLMoE-1B-7B-0924-a3s /models/OLMoE-1B-7B-0924-a3s-encrypted \
  --key-env A3S_MOE_WEIGHT_KEY --chunk-mib 1 \
  > olmoe-encryption.json
```

将发出的`manifestSha256`存储为检查点的带外信任
锚。密集且专业的权重被加密； `config.json`、`manifest.json`、
可选的标记生成器仍然是明文，但受 SHA-256 约束
可信清单。加密和开放使用有界块并且从不发布
解密的中间文件。

通过 Power 的 OpenAI 兼容 API 提供打包检查点：

```shell
cargo run --release --features server --bin a3s-moe-server -- \
  /models/OLMoE-1B-7B-0924-a3s \
  --model olmoe-1b-7b --device cpu --host-cache-mib 512 \
  --max-concurrent-requests 4
```

服务器检测到`olmoe`、`qwen3_moe`或`qwen3_5_moe`
检查点的有界`config.json`，注入相应的类型化后端，并注册一个
进程本地清单。支持的采样控制有
`temperature`、`top_p`、`top_k`、`min_p`、`seed`、`repeat_penalty`、
`repeat_last_n`、`frequency_penalty`、`presence_penalty`。不支持
模式、工具、结构化输出、交叉请求 KV 会话和后端
旋钮在推理之前失败。

通过相同的二进制文件提供打包的 Qwen3-MoE 检查点；省略
`--model` 选择特定于架构的默认 `qwen3-moe` 标识符：

```shell
cargo run --release --features server --bin a3s-moe-server -- \
  /models/Qwen3-30B-A3B-Base-a3s --device cpu \
  --host-cache-mib 512 --max-concurrent-requests 4
```

对于 Qwen3.6，省略 `--model` 选择 `qwen3.6-35b-a3b`：

```shell
cargo run --release --features server --bin a3s-moe-server -- \
  /models/Qwen3.6-35B-A3B-a3s --device cpu \
  --host-cache-mib 4096 --max-concurrent-requests 4
```

加密服务加载目前仅适用于OLMoE机密
检查点信封，并且 Qwen3-MoE 和 Qwen3.6 关闭失败。

通过提供其固定的信任锚和
拥有密钥的环境变量：

```shell
cargo run --release --features server --bin a3s-moe-server -- \
  /models/OLMoE-1B-7B-0924-a3s-encrypted --model olmoe-1b-7b \
  --encrypted-manifest-sha256 <manifestSha256> \
  --encrypted-key-env A3S_MOE_WEIGHT_KEY
```

生产机密主机应构建类型化加密源
来自他们经过验证的钥匙释放机制。基于环境的 CLI 是一个
避免命令行密钥暴露的操作员边界；它本身不是一个
远程认证协议。

相同的图可以在功率解析加速器上运行。准确构建一个
平台功能并显式选择类型设备：

```shell
cargo run --release --features server,cuda --bin a3s-moe-server -- \
  /models/OLMoE-1B-7B-0924-a3s --device cuda:0 \
  --host-cache-mib 512 --device-cache-mib 4096
```

Metal 在 macOS 上使用 `--features server,metal --device metal:0`。明确的
如果后端或序号不可用，则 CUDA 或 Metal 请求将失败。 `auto`
是唯一允许回退到 CPU 的模式，并且已解析的设备加上
后备决策作为无内容服务/基准证据公开。

生成原始的、可重复的性能证据，并可选择与
独立子进程中完全驻留的 CPU 路径：

```shell
cargo run --release --features benchmark --bin a3s-moe-bench -- \
  /models/OLMoE-1B-7B-0924-a3s \
  --prompt "Bitcoin is" --max-tokens 32 --warm-samples 5 \
  --checkpoint-label olmoe-1b-7b-bf16 \
  --resident-checkpoint /models/OLMoE-1B-7B-0924 \
  > olmoe-performance.json
```

Qwen3-MoE 使用相同的工具和特定于家庭的证据模式。其
公共奇偶校验门是固定的独立 F32 操作预言机
BF16权重，所以故意省略了内存密集型常驻F32
孩子：

```shell
cargo run --release --features benchmark --bin a3s-moe-bench -- \
  /models/Qwen3-30B-A3B-Base-a3s \
  --prompt "Bitcoin is" --max-tokens 8 --warm-samples 3 \
  --host-cache-mib 4096 --checkpoint-label qwen3-30b-a3b-bf16 \
  > qwen3-moe-performance.json
```

Qwen3.6 使用相同的证据边界和自己的版本化模式：

```shell
cargo run --release --features benchmark --bin a3s-moe-bench -- \
  /models/Qwen3.6-35B-A3B-a3s \
  --prompt "Bitcoin is" --max-tokens 8 --warm-samples 3 \
  --host-cache-mib 4096 --checkpoint-label qwen3.6-35b-a3b-bf16 \
  > qwen3.6-35b-a3b-performance.json
```

检查的 CUDA 运行使用显式设备请求和有界设备缓存：

```shell
cargo run --release --features benchmark,cuda --bin a3s-moe-bench -- \
  /models/Qwen3.6-35B-A3B-a3s \
  --prompt "Hello" --max-tokens 16 --warm-samples 3 \
  --device cuda:0 --host-cache-mib 0 --device-cache-mib 8192 \
  --checkpoint-label qwen3.6-35b-a3b-bf16 \
  > qwen3.6-35b-a3b-cuda-performance.json
```

第一个示例以空的 Power Expert 缓存开始。温暖的样品保留
仅配置的有界缓存。报告明确指出了运营
系统页面缓存不受控制；它并不称这种情况为身体状况
冷 I/O。检查的 Qwen3.6 CPU 工件报告`0.194616 tokens/s` 温暖
意思是；检查的 CUDA 工件报告了 `0.263611 tokens/s` 暖平均值。参见
[Performance Evidence](docs/performance.md) 为测量边界，
比较规则。

CUDA 和 Metal 是互为平台特定的 Cargo 功能，因此可移植 CI
使用 `--features server,benchmark,validation` 而不是 `--all-features`。
CUDA 和 macOS 运行器必须分别编译和测试 `cuda` 和 `metal`。

仅当有意更改固定模型时才重新生成微型预言机
合同：

```shell
python tools/generate_tiny_oracle.py
```

生成器将 JSON 打印到标准输出，并且永远不会覆盖已检查的装置。
在替换 `tests/fixtures/olmoe_tiny_oracle.json` 之前查看更改。
同样的规则也适用于 `generate_full_model_oracle.py` 及其完整的
解码器夹具。

Qwen3-MoE 稀疏层和全解码器装置遵循相同的
仅审阅工作流程：

```shell
python tools/generate_qwen3_moe_oracle.py
python tools/generate_qwen3_moe_full_oracle.py
```

M7现在提供常驻CPU参考推理，严格的官方拆分和
融合导出器检查点加载、分词器集成、有界转换、
强大的专家流媒体、路线联合连续批处理、服务
组成，并为第二个家庭固定了公众接受工具。它
重用 Power 的模型中立`RoutedExpertBatch`、经过验证的张量范围 I/O，
生命周期和唯一的居住层次结构。固定的公共模型数值，
并发HTTP，性能报告在`evidence/`下检查，所以
CPU Qwen3-MoE 部署路径在修订版 `0146bd6` 中接受 Power
修订`42c6646`。

## 许可证

MIT
