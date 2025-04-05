<p align="center">
  <img src="/Llama_Repo.jpeg" width="400"/>
</p>

<p align="center">
        🤗 <a href="https://huggingface.co/meta-Llama"> Models on Hugging Face</a>&nbsp | <a href="https://ai.meta.com/blog/"> Blog</a>&nbsp |  <a href="https://llama.meta.com/">Website</a>&nbsp | <a href="https://llama.meta.com/get-started/">Get Started</a>&nbsp | <a href="https://github.com/meta-llama/llama-cookbook">Llama Cookbook</a>&nbsp
<br>

---

# Llama Models

Llama is an accessible, open large language model (LLM) designed for developers, researchers, and businesses to build, experiment, and responsibly scale their generative AI ideas. Part of a foundational system, it serves as a bedrock for innovation in the global community. A few key aspects:
1. **Open access**: Easy accessibility to cutting-edge large language models, fostering collaboration and advancements among developers, researchers, and organizations
2. **Broad ecosystem**: Llama models have been downloaded hundreds of millions of times, there are thousands of community projects built on Llama and platform support is broad from cloud providers to startups - the world is building with Llama!
3. **Trust & safety**: Llama models are part of a comprehensive approach to trust and safety, releasing models and tools that are designed to enable community collaboration and encourage the standardization of the development and usage of trust and safety tools for generative AI

Our mission is to empower individuals and industry through this opportunity while fostering an environment of discovery and ethical AI advancements. The model weights are licensed for researchers and commercial entities, upholding the principles of openness.

## Llama Models

[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-models)](https://pypi.org/project/llama-models/)
[![Discord](https://img.shields.io/discord/1257833999603335178)](https://discord.gg/TZAAYNVtrU)

|  **Model** | **Launch date** | **Model sizes** | **Context Length** | **Tokenizer** | **Acceptable use policy**  |  **License** | **Model Card** |
| :----: | :----: | :----: | :----:|:----:|:----:|:----:|:----:|
| Llama 2 | 7/18/2023 | 7B, 13B, 70B | 4K | Sentencepiece | [Use Policy](models/llama2/USE_POLICY.md) | [License](models/llama2/LICENSE) | [Model Card](models/llama2/MODEL_CARD.md) |
| Llama 3 | 4/18/2024 | 8B, 70B | 8K | TikToken-based | [Use Policy](models/llama3/USE_POLICY.md) | [License](models/llama3/LICENSE) | [Model Card](models/llama3/MODEL_CARD.md) |
| Llama 3.1 | 7/23/2024 | 8B, 70B, 405B | 128K | TikToken-based | [Use Policy](models/llama3_1/USE_POLICY.md) | [License](models/llama3_1/LICENSE) | [Model Card](models/llama3_1/MODEL_CARD.md) |
| Llama 3.2 | 9/25/2024 | 1B, 3B | 128K | TikToken-based | [Use Policy](models/llama3_2/USE_POLICY.md) | [License](models/llama3_2/LICENSE) | [Model Card](models/llama3_2/MODEL_CARD.md) |
| Llama 3.2-Vision | 9/25/2024 | 11B, 90B | 128K | TikToken-based | [Use Policy](models/llama3_2/USE_POLICY.md) | [License](models/llama3_2/LICENSE) | [Model Card](models/llama3_2/MODEL_CARD_VISION.md) |
| Llama 3.3 | 12/04/2024 | 11B, 90B | 128K | TikToken-based | [Use Policy](models/llama3_3/USE_POLICY.md) | [License](models/llama3_3/LICENSE) | [Model Card](models/llama3_3/MODEL_CARD.md) |
| Llama 4 | 4/5/2025 | Scout-17B-16E, Maverick-17B-128E | 10M, 1M | TikToken-based | [Use Policy](models/llama4/USE_POLICY.md) | [License](models/llama4/LICENSE) | [Model Card](models/llama4/MODEL_CARD.md) |

## Download

To download the model weights and tokenizer:

1. Visit the [Meta Llama website](https://llama.meta.com/llama-downloads/).
2. Read and accept the license.
3. Once your request is approved you will receive a signed URL via email.
4. Install the [Llama CLI](https://github.com/meta-llama/llama-stack): `pip install llama-stack`. (**<-- Start Here if you have received an email already.**)
5. Run `llama model list` to show the latest available models and determine the model ID you wish to download. **NOTE**:
If you want older versions of models, run `llama model list --show-all` to show all the available Llama models.

6. Run: `llama download --source meta --model-id CHOSEN_MODEL_ID`
7. Pass the URL provided when prompted to start the download.

Remember that the links expire after 24 hours and a certain amount of downloads. You can always re-request a link if you start seeing errors such as `403: Forbidden`.

## Running the models

In order to run the models, you will need to install dependencies after checking out the repository.

```bash
# Run this within a suitable Python environment (uv, conda, or virtualenv)
pip install -e .[torch]
```

Example scripts are available in `models/{ llama3, llama4 }/scripts/` sub-directory. Note that the Llama4 series of models require at least 4 GPUs to run inference at full (bf16) precision.

```bash
#!/bin/bash

NGPUS=4
CHECKPOINT_DIR=~/.llama/checkpoints/Llama-4-Scout-17B-16E-Instruct
PYTHONPATH=$(git rev-parse --show-toplevel) \
  torchrun --nproc_per_node=$NGPUS \
  -m models.llama4.scripts.chat_completion $CHECKPOINT_DIR \
  --world_size $NGPUS
```

The above script should be used with an Instruct (Chat) model. For a Base model, update the `CHECKPOINT_DIR` path and use the script `models.llama4.scripts.completion`.


## Running inference with FP8 and Int4 Quantization

You can reduce the memory footprint of the models at the cost of minimal loss in accuracy by running inference with FP8 or Int4 quantization. Use the `--quantization-mode` flag to specify the quantization mode. There are two modes:
- `fp8_mixed`: Mixed precision inference with FP8 for some weights and bfloat16 for activations.
- `int4_mixed`: Mixed precision inference with Int4 for some weights and bfloat16 for activations.

Using FP8, running Llama-4-Scout-17B-16E-Instruct requires 2 GPUs with 80GB of memory. Using Int4, you need a single GPU with 80GB of memory.

```bash
MODE=fp8_mixed  # or int4_mixed
if [ $MODE == "fp8_mixed" ]; then
  NGPUS=2
else
  NGPUS=1
fi
CHECKPOINT_DIR=~/.llama/checkpoints/Llama-4-Scout-17B-16E-Instruct
PYTHONPATH=$(git rev-parse --show-toplevel) \
  torchrun --nproc_per_node=$NGPUS \
  -m models.llama4.scripts.chat_completion $CHECKPOINT_DIR \
  --world_size $NGPUS \
  --quantization-mode $MODE
```


For more flexibility in running inference (including using other providers), please see the [`Llama Stack`](https://github.com/meta-llama/llama-stack) toolset.


## Access to Hugging Face

We also provide downloads on [Hugging Face](https://huggingface.co/meta-llama), in both transformers and native `llama3` formats. To download the weights from Hugging Face, please follow these steps:

- Visit one of the repos, for example [meta-llama/Llama-4-Scout-17B-16E](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E).
- Read and accept the license. Once your request is approved, you'll be granted access to all Llama 3.1 models as well as previous versions. Note that requests used to take up to one hour to get processed.
- To download the original native weights to use with this repo, click on the "Files and versions" tab and download the contents of the `original` folder. You can also download them from the command line if you `pip install huggingface-hub`:

```bash
huggingface-cli download meta-llama/Llama-4-Scout-17B-16E-Instruct-Original --local-dir meta-llama/Llama-4-Scout-17B-16E-Instruct-Original
```

- To use with transformers, the following [pipeline](https://huggingface.co/docs/transformers/en/main_classes/pipelines) snippet will download and cache the weights:

  ```python
  #inference.py
  from transformers import AutoTokenizer, Llama4ForConditionalGeneration
  import torch

  model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

  tokenizer = AutoTokenizer.from_pretrained(model_id)

  messages = [
      {"role": "user", "content": "Who are you?"},
  ]
  inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

  model = Llama4ForConditionalGeneration.from_pretrained(
      model_id,
      device_map="auto",
      torch_dtype=torch.bfloat16
  )

  outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
  outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])
  print(outputs[0])
  ```
  ```bash
   torchrun --nnodes=1 --nproc_per_node=8 inference.py
   ```

## Installations

You can install this repository as a [package](https://pypi.org/project/llama-models/) by just doing `pip install llama-models`

## Responsible Use

Llama models are a new technology that carries potential risks with use. Testing conducted to date has not — and could not — cover all scenarios.
To help developers address these risks, we have created the [Responsible Use Guide](https://ai.meta.com/static-resource/responsible-use-guide/).

## Issues

Please report any software “bug” or other problems with the models through one of the following means:
- Reporting issues with the model: [https://github.com/meta-llama/llama-models/issues](https://github.com/meta-llama/llama-models/issues)
- Reporting risky content generated by the model: [developers.facebook.com/llama_output_feedback](http://developers.facebook.com/llama_output_feedback)
- Reporting bugs and security concerns: [facebook.com/whitehat/info](http://facebook.com/whitehat/info)


## Questions

For common questions, the FAQ can be found [here](https://llama.meta.com/faq), which will be updated over time as new questions arise.
