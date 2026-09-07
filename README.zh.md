<p align="center">
  <img src="/Llama_Repo.jpeg" width="400"/>
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<p align="center">
        🤗 <a href="https://huggingface.co/meta-Llama">Hugging Face 上的模型</a>&nbsp | <a href="https://ai.meta.com/blog/">官方博客</a>&nbsp |  <a href="https://llama.meta.com/">官方网站</a>&nbsp | <a href="https://llama.meta.com/get-started/">快速入门</a>&nbsp | <a href="https://github.com/meta-llama/llama-cookbook">Llama Cookbook</a>&nbsp
</p>

---

# Llama 模型库 (Llama Models)

Llama 是一款触手可及的开放大语言模型（LLM），专为开发者、研究人员和企业打造，助力用户构建、试验并以负责任的方式规模化实现其生成式 AI 创意。作为基础系统的核心组成部分，它是全球开源社区创新的坚实基石。核心亮点包括：
1. **开放获取（Open access）**：便捷获取前沿尖端的大语言模型，促进开发者、科研人员与各类机构之间的深度协作与技术突破。
2. **广泛生态（Broad ecosystem）**：Llama 系列模型已被下载数亿次，全球已有数千个基于 Llama 构建的社区项目，并获得了从各大云厂商到初创企业的全面平台支持——全世界都在基于 Llama 进行构建！
3. **信任与安全（Trust & safety）**：Llama 模型贯彻全方位信任与安全理念，发布经过专门设计的模型与工具套件，赋能社区协同创新，并推动生成式 AI 信任与安全开发工具及使用规范的标准化。

我们的使命是通过这一机遇赋能个人与行业，同时营造鼓励探索与合乎伦理的 AI 发展环境。模型权重对研究人员和商业实体开放授权，坚守开放透明的核心原则。

## Llama 模型列表 (Llama Models)

[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-models)](https://pypi.org/project/llama-models/)
[![Discord](https://img.shields.io/discord/1257833999603335178)](https://discord.gg/TZAAYNVtrU)

|  **模型** | **发布日期** | **模型尺寸** | **上下文长度** | **分词器** | **可接受使用政策**  |  **许可证** | **模型卡片 (Model Card)** |
| :----: | :----: | :----: | :----:|:----:|:----:|:----:|:----:|
| Llama 2 | 2023/7/18 | 7B, 13B, 70B | 4K | Sentencepiece | [使用政策](models/llama2/USE_POLICY.md) | [许可证](models/llama2/LICENSE) | [模型卡片](models/llama2/MODEL_CARD.md) |
| Llama 3 | 2024/4/18 | 8B, 70B | 8K | 基于 TikToken | [使用政策](models/llama3/USE_POLICY.md) | [许可证](models/llama3/LICENSE) | [模型卡片](models/llama3/MODEL_CARD.md) |
| Llama 3.1 | 2024/7/23 | 8B, 70B, 405B | 128K | 基于 TikToken | [使用政策](models/llama3_1/USE_POLICY.md) | [许可证](models/llama3_1/LICENSE) | [模型卡片](models/llama3_1/MODEL_CARD.md) |
| Llama 3.2 | 2024/9/25 | 1B, 3B | 128K | 基于 TikToken | [使用政策](models/llama3_2/USE_POLICY.md) | [许可证](models/llama3_2/LICENSE) | [模型卡片](models/llama3_2/MODEL_CARD.md) |
| Llama 3.2-Vision | 2024/9/25 | 11B, 90B | 128K | 基于 TikToken | [使用政策](models/llama3_2/USE_POLICY.md) | [许可证](models/llama3_2/LICENSE) | [视觉模型卡片](models/llama3_2/MODEL_CARD_VISION.md) |
| Llama 3.3 | 2024/12/04 | 70B | 128K | 基于 TikToken | [使用政策](models/llama3_3/USE_POLICY.md) | [许可证](models/llama3_3/LICENSE) | [模型卡片](models/llama3_3/MODEL_CARD.md) |
| Llama 4 | 2025/4/5 | Scout-17B-16E, Maverick-17B-128E | 10M, 1M | 基于 TikToken | [使用政策](models/llama4/USE_POLICY.md) | [许可证](models/llama4/LICENSE) | [模型卡片](models/llama4/MODEL_CARD.md) |

## 模型下载 (Download)

如需下载模型权重与分词器，请按以下步骤操作：

1. 访问 [Meta Llama 官方网站](https://llama.meta.com/llama-downloads/)。
2. 阅读并接受许可协议。
3. 申请审核通过后，你将通过电子邮件收到一个专属签名下载链接（Signed URL）。
4. 安装 Llama Models CLI 命令行工具：`pip install llama-models`。（**<-- 如果你已经收到许可邮件，可直接从本步骤开始。**）
5. 运行 `llama-model list` 查看最新可用模型，并确定你想下载的模型 ID。**提示**：
若需查看历史版本的模型，请运行 `llama-model list --show-all` 列出所有可用的 Llama 模型。
6. 运行下载命令：`llama-model download --source meta --model-id 目标模型ID`
7. 提示输入链接时，粘贴邮件中收到的专属签名 URL 即可开始下载。

请注意：专属签名链接将在 24 小时后或达到指定下载次数限制后失效。如果遇到 `403: Forbidden` 等错误，可随时重新申请下载链接。

### CLI 命令行工具参考 (CLI Commands Reference)

安装后，`llama-model` CLI 提供了以下常用命令：

```bash
llama-model list              # 列出当前可用模型
llama-model list --show-all   # 列出所有模型（包含历史旧版本）
llama-model describe -m MODEL_ID     # 查看指定模型的详细元数据与配置
llama-model download          # 从 Meta 或 Hugging Face 下载模型
llama-model verify-download   # 校验已下载模型的完整性
llama-model remove -m MODEL_ID       # 删除已下载的本地模型
llama-model prompt-format -m MODEL_ID  # 显示该模型的 Prompt 格式规范
```

如需查看任意命令的详细帮助手册，可运行 `llama-model COMMAND --help`。

## 运行模型 (Running the models)

克隆并检出本仓库后，需要先安装运行依赖：

```bash
# 请在合适的 Python 环境（如 uv、conda 或 virtualenv）中执行
pip install .[torch]
```

示例脚本位于 `models/{ llama3, llama4 }/scripts/` 子目录下。请注意：以全精度（bf16）运行 Llama 4 系列模型推理至少需要 4 张 GPU。

```bash
#!/bin/bash

NGPUS=4
CHECKPOINT_DIR=~/.llama/checkpoints/Llama-4-Scout-17B-16E-Instruct
PYTHONPATH=$(git rev-parse --show-toplevel)   torchrun --nproc_per_node=$NGPUS   -m models.llama4.scripts.chat_completion $CHECKPOINT_DIR   --world_size $NGPUS
```

上述脚本适用于 Instruct（对话）模型。若运行 Base 基础模型，请修改 `CHECKPOINT_DIR` 路径并使用 `models.llama4.scripts.completion` 脚本。

## 使用 FP8 与 Int4 量化进行推理 (Running inference with FP8 and Int4 Quantization)

通过 FP8 或 Int4 量化推理，可以在精度损失极小的前提下显著降低模型的显存占用。使用 `--quantization-mode` 参数即可指定量化模式，支持以下两种模式：
- `fp8_mixed`：混合精度推理，部分权重使用 FP8，激活值保留 bfloat16。
- `int4_mixed`：混合精度推理，部分权重使用 Int4，激活值保留 bfloat16。

使用 FP8 时，运行 Llama-4-Scout-17B-16E-Instruct 需要 2 张 80GB 显存的 GPU。使用 Int4 时，仅需单张 80GB 显存的 GPU 即可运行。

```bash
MODE=fp8_mixed  # 或 int4_mixed
if [ $MODE == "fp8_mixed" ]; then
  NGPUS=2
else
  NGPUS=1
fi
CHECKPOINT_DIR=~/.llama/checkpoints/Llama-4-Scout-17B-16E-Instruct
PYTHONPATH=$(git rev-parse --show-toplevel)   torchrun --nproc_per_node=$NGPUS   -m models.llama4.scripts.chat_completion $CHECKPOINT_DIR   --world_size $NGPUS   --quantization-mode $MODE
```

若需更灵活的推理运行方案（包括接入其他推理后端与服务商），请参考 [`Llama Stack`](https://github.com/meta-llama/llama-stack) 工具套件。

## 从 Hugging Face 获取 (Access to Hugging Face)

我们在 [Hugging Face](https://huggingface.co/meta-llama) 上同步提供了权重下载，同时支持 transformers 格式与原生 `llama4` 格式。如需从 Hugging Face 下载权重，请按照以下步骤操作：

- 访问对应的模型仓库页面，例如 [meta-llama/Llama-4-Scout-17B-16E](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E)。
- 阅读并签署许可协议。审核通过后，你将获得所有 Llama 3.1 模型及先前版本的访问权限（此前审核处理可能需要最长 1 小时）。
- 若要下载配合本仓库使用的原始原生格式权重，请点击 "Files and versions" 标签页并下载 `original` 文件夹中的内容。你也可以先安装 `pip install huggingface-hub`，然后直接通过命令行下载：

```bash
huggingface-cli download meta-llama/Llama-4-Scout-17B-16E-Instruct-Original --local-dir meta-llama/Llama-4-Scout-17B-16E-Instruct-Original
```

- 若要与 transformers 配合使用，可通过以下代码片段自动下载并缓存权重：

  ```python
  # inference.py
  from transformers import AutoTokenizer, Llama4ForConditionalGeneration
  import torch

  model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

  tokenizer = AutoTokenizer.from_pretrained(model_id)

  messages = [
      {"role": "user", "content": "Who are you?"},
  ]
  inputs = tokenizer.apply_chat_template(
      messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
  )

  model = Llama4ForConditionalGeneration.from_pretrained(
      model_id, device_map="auto", torch_dtype=torch.bfloat16
  )

  outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
  outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1] :])
  print(outputs[0])
  ```
  ```bash
  torchrun --nnodes=1 --nproc_per_node=8 inference.py
  ```

## 安装方式 (Installations)

你可以直接将本仓库作为 Python [软件包](https://pypi.org/project/llama-models/) 进行安装：
```bash
pip install llama-models
```

## 负责任使用 (Responsible Use)

Llama 模型是一项新兴技术，在使用过程中存在潜在风险。迄今为止进行的安全测试无法且不可能覆盖所有应用场景。
为协助开发者积极防范和应对这些风险，我们编写了 [负责任使用指南 (Responsible Use Guide)](https://ai.meta.com/static-resource/responsible-use-guide/)。

## 问题反馈 (Issues)

如发现任何软件缺陷（Bug）或模型运行异常，请通过以下途径反馈：
- 报告模型相关问题：[https://github.com/meta-llama/llama-models/issues](https://github.com/meta-llama/llama-models/issues)
- 报告模型生成的不安全/风险内容：[developers.facebook.com/llama_output_feedback](http://developers.facebook.com/llama_output_feedback)
- 报告代码安全漏洞与缺陷：[facebook.com/whitehat/info](http://facebook.com/whitehat/info)

## 常见问题 (Questions)

关于常见问题解答，请参阅官方 [FAQ](https://llama.meta.com/faq)，该页面会随着新问题的出现持续同步更新。

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年9月7日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
