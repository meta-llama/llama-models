
# Llama 3 Evaluation Details

This document contains some additional context on the settings and methodology for how we evaluated the Llama 3.1 8B, 70B, and 405B pre-trained and post-trained models.


## Language auto-eval benchmark notes:

For a given benchmark, we strive to use consistent evaluation settings across all models, including external models. We make every effort to achieve optimal scores for external models, including addressing any model-specific parsing and tokenization requirements. Where the scores are lower for external models than self-reported scores on comparable or more conservative settings, we report the self-reported scores for external models. We are also releasing the data generated as part of evaluations with publicly available benchmarks which can be found on [huggingface here](https://huggingface.co/meta-llama).


### MMLU

For the pre-trained models we use a 5-shot config. To determine the choice character we use the standard MMLU prompt and compare the negative log-likelihood (NLL) of the various choices.

For the post-trained models we report both 5-shot and 0-shot scores. We ask the model to generate the best choice character. The 0-shot scores use a CoT (chain of thought) prompt. The maximum generation lengths for the 5-shot and 0-shot configs are 10 tokens and 1024 tokens respectively.

Macro averages are reported unless otherwise stated. The micro average scores for the various models are: 65.6, 79.0, and 85.4 for the pre-trained 8B, 70B and 405B models respectively for the 5-shot config; 69.44, 84.0, 87.71 for the post-trained 8B, 70B and 405B models respectively for the 5-shot config.


### MMLU-Pro

For the pre-trained and post-trained models we use a 5-shot config with CoT prompt. We ask the model to generate the reasoning and the corresponding best choice character. The maximum generation length is 512 tokens for pre-trained setup and 1024 for post-trained setup.


### ARC-Challenge

We use the Arc-Challenge subset from the Arc benchmark. For the pre-trained models, we use a  25-shot config and use the MMLU setup for evaluation where we provide all the choices in the prompt and calculate likelihood over choice characters. For the post-trained models, we use 0-shot config and ask the model to generate the choice character. The maximum generation length is 100 tokens.


### GPQA

For post-trained models, we use 0-shot config with and without CoT prompt and report exact match scores over the possible options using the main set. Maximum generation length is 96 tokens when not using CoT prompt and 2048 tokens when using the CoT prompt.


### AGIEval English

For pre-trained models, we use the default few-shot and prompt settings as specified [here](https://github.com/ruixiangcui/AGIEval). The score is averaged over the english subtasks. The maximum generation length is 10 tokens.


### IFEval

For post-trained models, we use the default settings as specified [here](https://arxiv.org/pdf/2311.07911). We compute the prompt level scores and instruction level strict and loose accuracy. We then report the average across all the scores.


### HumanEval/HumanEval+

For both pre-trained and post-trained models, we use a 0-shot config and report pass@1 scores. The maximum generation length is 1024 tokens.


### CommonSenseQA

For pre-trained models, we use the same 7-shot config with CoT prompt as in [Wei et al. (2022)](https://arxiv.org/pdf/2201.11903.pdf). We use the MMLU setup for evaluation where we provide all the choices in the prompt and calculate likelihood over choice characters.


### WinoGrande

For pre-trained models, we use a choice based setup for evaluation where we fill in the missing blank with the two possible choices and then compute log-likelihood over the suffix. We use a 5-shot config. We use the MMLU setup where we provide all the choices in the prompt and calculate likelihood over choice characters.


### BIG-Bench Hard

For pre-trained models, we use a 3-shot config with CoT prompt and compute the average exact match over the subsets in this task. We run this as a generative task. Maximum generation length is 512 tokens.


### SQuAD

For pre-trained models, we use SQuAD v2 with a 1-shot config and report exact match scores. We run this as a generative task. Maximum generation length is 32 tokens.


### QuAC

For pre-trained models, we use a 1-shot config and report the F1 scores. We run this as a generative task. Maximum generation length is 32 tokens.


### BoolQ

For pre-trained models, we use a 0-shot config and report average accuracy. We run this as a choice task.


### DROP

For pre-trained models, for each validation example, we draw 3 random few-shot examples from the train split and report the F1 scores. The maximum generation length is 32 tokens.


### GSM8K

For both pre-trained and post-trained models, we use the same 8-shot config with CoT prompt as in [Wei et al. (2022)](https://arxiv.org/pdf/2201.11903.pdf) (maj@1). The maximum generation length is 1024 tokens.


### RACE

For pre-trained models, we use a 0-shot config. We run this as a choice task. We use the MMLU setup for evaluation where we provide all the choices in the prompt and calculate likelihood over choice characters.


### WorldSense

For pre-trained models, we use a 0-shot config. We run this as a choice task. Unlike the original benchmark, we do not normalize the three-option partitions of the benchmark. The chance accuracy is therefore not 0.5, but averages to 0.46.


### MBPP

For pre-trained and post-trained models we use a 3-shot config and report pass@1 scores. We run this as a generative task. Maximum generation length is 256 tokens.


### MBPP EvalPlus (base)

For pre-trained and post-trained models we use a 0-shot config and report pass@1 scores. We run this as a generative task. Maximum generation length is 1024 tokens.


### MATH

For pre-trained models, we use the same 4-shot config as in [Lewkowycz et al. (2022)](https://arxiv.org/pdf/2206.14858.pdf) (maj@1). Maximum generation length is 512 tokens.

For post-trained models we use a 0-shot config with Cot prompt. We enhance the exact match using [sympy](https://www.sympy.org/en/index.html) and then use an [equality template](https://github.com/openai/simple-evals/blob/main/common.py#L27-L85) with a judge to resolve complex expressions. Maximum generation length is 5120 tokens. The MATH score represents the full dataset. The scores for MATH-HARD (Lvl 5) are 25.4, 43.8, and 53.4 for the 8B, 70B and 405B models respectively.


### SCROLLS

For pre-trained models, we use a 5-shot config. Maximum generation length is 32 tokens. Maximum input prompt length is 131072 less the number of tokens generated (i.e. 131040).


### ZeroSCROLLS

For post-trained models, we use a 0-shot config. Maximum generation length for QuALITY and SQuALITY is 64 tokens. For Qasper it is 128 tokens. Maximum input prompt length for Llama models is 131072 less the number of tokens generated for each task (i.e. 131008 for QuALITY and SQuALITY and 130944 for Qasper). Maximum input length for non-llama models is 128000 less the number of tokens generated for each task. We ensure that all relevant information is retained in the context for all models for fair comparison.


### InfiniteBench

For post-trained models, we use a 0-shot config. Maximum generation length is 20 for both the En.QA and En.MC tasks and maximum input prompt length is 131052. Maximum input length for non-llama models is 127980. We ensure that all relevant information is retained in the context for all models for fair comparison.


### NIH/Multi-needle

For post-training, we use a 0-shot config. Our context lengths are evenly spaced between 2000 and 131072 in 10 intervals, inclusive of the endpoints for llama models and between 2000 and 128000 for non-llama models. Maximum generation length is 256 tokens.


### Multilingual MGSM

For post-trained models, we use an 0-shot config with CoT prompt and report exact match (maj@1) scores. Maximum generation length is 2048 tokens. The scores are averaged over all the eleven languages present in the MGSM benchmark, including the ones not supported by Llama models.


### Multilingual MMLU

For post-trained models, we use a 5-shot config. We run this as a generative task. Maximum generation length is 10 tokens. The scores are individually reported for each and averaged over the seven non-english languages that Llama models support (Portuguese, Spanish, Italian, German, French, Hindi, Thai).


### Multipl-E HumanEval and Multipl-E MBPP

For post-trained models, we use a 0-shot config and report pass@1 scores. Maximum generation length is 512 tokens. Where Multipl-E average is reported, the scores are averaged over all 6 languages in the benchmark.


### PiQA, SiQA, and OpenBookQA

For pre-trained models, we use a 0-shot config and report average accuracy. We run these as choice task.


### Dynabench SQuAD and Adversarial SQuAD

For the adversarial versions of squad ([Dynabench](https://aclanthology.org/2021.naacl-main.324/) and [Adversarial](https://aclanthology.org/D17-1215/)), we use the same setting as standard SQuAD (1-shot config and exact match as the metric)


### PAWS

For pre-trained models, we use a 5-shot config and report exact match scores. We run this as a generative task. Maximum generation length is 32 tokens.


### GSM Plus

For pre-trained models, we use the same 8-shot config with CoT prompt as in [Wei et al. (2022)](https://arxiv.org/pdf/2201.11903.pdf) (maj@1). The maximum generation length is 512 tokens.


### Berkeley Function Calling Leaderboard (BFCL)

Benchmark results were achieved by running the open source evaluation repository [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) on commit 7bef000 without any further changes.


### Nexus

We use the [open-source ](https://github.com/nexusflowai/NexusRaven)prompt and evaluation function followed by the[ open source notebook](https://github.com/nexusflowai/NexusRaven-V2/blob/master/evaluation_notebook/GPT4_Evaluation/Benchmark_GPT4.ipynb) to compute the scores.


### API Bank

We use a 0-shot config with a custom prompt and parsing function to reduce the incidence of false negatives. We also modify the dataset by correcting and completing the ground truth answers that were initially incorrect or incomplete. Second, we improve the evaluation metric to better assess function call correctness by splitting keyword arguments into two groups. We use exact match for keyword arguments that have a unique ground truth, and ROUGE score for those that accept any string with the same semantic meaning as the reference value.

### Gorilla API Bench

For post-trained models, we use the same 0-shot prompt and evaluation function as proposed in the [original paper](https://arxiv.org/abs/2305.15334). Just like the [open-source](https://github.com/ShishirPatil/gorilla) implementation, we compare the domains of the retrieved API call from the API database with the ground truth. If the domain of the retrieved API is the same as the ground truth and the API exists in the database, it is considered a success. All other scenarios are considered failures.

### TriviaQA-WIKI
For TrivialQA, we evaluate on the Wiki validation set, use 5-shot config and compute average exact match. We run this as a generative task. Maximum generation length is 24 tokens.
