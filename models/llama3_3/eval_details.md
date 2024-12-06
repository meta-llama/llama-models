

# Llama 3 Evaluation Details

This document contains some additional context on the settings and methodology for how we evaluated the Llama 3.2 models.


## Language auto-eval benchmark notes:

For a given benchmark, we strive to use consistent evaluation settings across all models, including external models. We make every effort to achieve optimal scores for external models, including addressing any model-specific parsing and tokenization requirements. Where the scores are lower for external models than self-reported scores on comparable or more conservative settings, we report the self-reported scores for external models. We are also releasing the data generated as part of evaluations with publicly available benchmarks which can be found on [huggingface here](https://huggingface.co/collections/meta-llama/llama-32-evals-66f44b3d2df1c7b136d821f0).


### MMLU

For the pre-trained models we use a 5-shot config. To determine the choice character we use the standard MMLU prompt and compare the negative log-likelihood (NLL) of the various choices.

For the post-trained models we report both 5-shot and 0-shot scores. We ask the model to generate the best choice character. The 0-shot scores use a CoT (chain of thought) prompt. The maximum generation lengths for the 5-shot and 0-shot configs are 10 tokens and 1024 tokens respectively.

Macro averages are reported unless otherwise stated. The micro average scores for the various models are: 65.6, 79.0, and 85.4 for the pre-trained 8B, 70B and 405B models respectively for the 5-shot config; 69.44, 84.0, 87.71 for the post-trained 8B, 70B and 405B models respectively for the 5-shot config.


### TLDR9+

For post-trained models, we use a 1-shot config and report rougeL scores. We run this as a generative task. Maximum generation length is 512 tokens. We specifically ran this on [TLDR9+ dataset](https://github.com/sajastu/reddit_collector)


### Open-Rewrite

For post-trained models, we use a 0-shot config and report micro_avg rougeL scores across elaborate, formality, others, paraphrase, shorten, and wiki. We run this as a generative task. Maximum generation length is 512 tokens. Specific dataset can be found [here](https://github.com/google-research/google-research/tree/master/rewritelm).


### IFEval

For post-trained models, we use the default settings as specified [here](https://arxiv.org/pdf/2311.07911). We compute the prompt level scores and instruction level strict and loose accuracy. We then report the average across all the scores.


### ARC-Challenge

We use the Arc-Challenge subset from the Arc benchmark. For the pre-trained models, we use a  25-shot config and use the MMLU setup for evaluation where we provide all the choices in the prompt and calculate likelihood over choice characters. For the post-trained models, we use 0-shot config and ask the model to generate the choice character. The maximum generation length is 100 tokens.


### GPQA

For post-trained models, we use 0-shot config with CoT prompt and report exact match scores over the possible options using the main set. Max generation length is 2048 tokens.


### GPQA Diamond

GPQA Diamond is the subset of GPQA where two out of two experts agree, Rein et al. (2023). For post-trained models, we use 0-shot config with CoT prompt and report exact match scores over the possible options using the Diamond set. Max generation length is 2048 tokens.

### AGIEval English

For pre-trained models, we use the default few-shot and prompt settings as specified [here](https://github.com/ruixiangcui/AGIEval). The score is averaged over the english subtasks. The max generation length is 10 tokens.


### SQuAD

For pre-trained models, we use SQuAD v2 with a 1-shot config and report exact match scores. We run this as a generative task. Maximum generation length is 32 tokens.


### QuAC

For pre-trained models, we use a 1-shot config and report the F1 scores. We run this as a generative task. Maximum generation length is 32 tokens.


### DROP

For pre-trained models, for each validation example, we draw 3 random few-shot examples from the train split and report the F1 scores. The maximum generation length is 32 tokens.


### GSM8K

For both pre-trained and post-trained models, we use the same 8-shot config with CoT prompt as in [Wei et al. (2022)](https://arxiv.org/pdf/2201.11903.pdf) (maj@1). The maximum generation length is 1024 tokens.


### MATH

For pre-trained models, we use the same 4-shot config as in [Lewkowycz et al. (2022)](https://arxiv.org/pdf/2206.14858.pdf) (maj@1). Maximum generation length is 512 tokens.

For post-trained models we use a 0-shot config with Cot prompt. We enhance the exact match using [sympy](https://www.sympy.org/en/index.html) and then use an [equality template](https://github.com/openai/simple-evals/blob/main/common.py#L27-L85) with a judge to resolve complex expressions. Maximum generation length is 5120 tokens. The MATH score represents the full dataset. The scores for MATH-HARD (Lvl 5) are 25.4, 43.8, and 53.4 for the 8B, 70B and 405B models respectively.


### InfiniteBench

We report on EN.MC (Free-form question answering based on the fake book) and EN.QA (Multiple choice questions derived from the fake book) sub-tasks. The average tokens in these tasks is 184.4k and 192.6k respectively. We truncate down the dataset to 128k input tokens using our tokenizer to have a 'clean' dataset where the right answer is not mistakenly deleted during truncation. For post-trained models, we use a 0-shot config. Maximum generation length is 20 for both the En.QA and En.MC tasks.


### NIH/Multi-needle

For post-training, we use a 0-shot config. Our context lengths are evenly spaced between 2000 and 131072 in 10 intervals, inclusive of the endpoints for llama models and between 2000 and 128000 for non-llama models. Maximum generation length is 256 tokens.


### MGSM

For post-trained models, we use an 0-shot config with CoT prompt and report exact match (maj@1) scores. Maximum generation length is 2048 tokens. The scores are averaged over all the eleven languages present in the MGSM benchmark, including the ones not supported by Llama models.


### Multilingual MMLU

For post-trained models, we use a 5-shot config. We run this as a generative task. Maximum generation length is 10 tokens. The scores are individually reported for each and averaged over the seven non-english languages that Llama models support (Portuguese, Spanish, Italian, German, French, Hindi, Thai).


### Berkeley Function Calling Leaderboard (BFCL) v2

For Berkeley Function Calling Leaderboard (BFCL-v2), benchmark results were achieved by running the open source evaluation repository [ShishirPatil/gorilla ](https://github.com/ShishirPatil/gorilla/)on commit 70d6722 and we report the "AST Summary" metric.


### Nexus

We use the [open-source ](https://github.com/nexusflowai/NexusRaven)prompt and evaluation function followed by the[ open source notebook](https://github.com/nexusflowai/NexusRaven-V2/blob/master/evaluation_notebook/GPT4_Evaluation/Benchmark_GPT4.ipynb) to compute the scores.


### RULER

For more comprehensive long-context evals beyond retrieval, we assess our perf on RULER benchmark, where we synthesize datasets across increasingly long context length buckets, and compare mean across each of them over retrieval (single needle, multi needle, multi-value per multiple keys), multi-hop tracing (variable tracking), aggregation (common words, frequency extraction), and question-answering.


### MMMU

For post-trained models, we use 0-shot config with CoT prompt and report scores over the possible options using the validation set. Maximum generation length is 2048 tokens. We use the following system prompt: "&lt;|image|>Look at the image carefully and solve the following question step-by-step. {question} Options: {options} Indicate the correct answer at the end."


### MMMU-Pro standard

For post-trained models, we use 0-shot config, we use multiple choice questions with ten options, and report scores over the possible options using the test set. In case of multiple images provided per prompt, they are stitched into a single image. Maximum generation length is 2048 tokens. We use the following system prompt: "&lt;|image|>{question} Options: {options}"


### MMMU-Pro vision

For post-trained models, we use 0-shot config and report scores over the possible options using the test set. Maximum generation length is 2048 tokens. We use the following system prompt: "&lt;|image|>Your job is to extract the question from the image the user attached, reason about it, and then answer the question. Follow these steps: \n1. Always start by Clearly stating the question shown in the image, and also state all of the options. \n2. Then, Carefully review the information beyond the question shown in the image and analyze it without making any assumptions. \n3. Next, use your understanding of the image to answer the question you listed out in the first step. Use a step-by-step process \n4. Finally, write the answer in the following format where X is exactly one of the option letters: The best answer is X.\nAlways follow the steps above, printing out everything. Let's think step by step. Your response must be among the given options."


### AI2D

For post-trained models, we use 0-shot config and report scores using the test set. Maximum generation length is 400 tokens. For 11b we use the following system prompt: "&lt;|image|>Look at the scientific diagram carefully and answer the following question: {question}\n Think step by step and finally respond to the question with only the correct option number as \"FINAL ANSWER\". Let's think step by step." For 90b we use a different system prompt: "&lt;|image|>Look at the scientific diagram carefully and answer the following question: {question}\n Respond only with the correct option digit."


### ChartQA

For post-trained models, we use 0-shot config with CoT prompt and report scores using the test set. Maximum generation length is 512 tokens. For 11b we use the following system prompt: "&lt;|image|>You are provided a chart image and will be asked a question. You have to think through your answer and provide a step-by-step solution. Once you have the solution, write the final answer in at most a few words at the end with the phrase "FINAL ANSWER:". The question is: {question}&lt;cot_start>Let's think step by step." For 90b we use a different system prompt: "&lt;|image|>You are provided a chart image and will be asked a question. Follow these steps carefully:\n Step 1: Analyze the question to understand what specific data or information is being asked for. Focus on whether the question is asking for a specific number or category from the chart image.\n Step 2: Identify any numbers, categories, or groups mentioned in the question and take note of them. Focus on detecting and matching them directly to the image. \nStep 3: Study the image carefully and find the relevant data corresponding to the categories or numbers mentioned. Avoid unnecessary assumptions or calculations; simply read the correct data from the image.\n Step 4: Develop a clear plan to solve the question by locating the right data. Focus only on the specific category or group that matches the question. \nStep 5: Use step-by-step reasoning to ensure you are referencing the correct numbers or data points from the image, avoiding unnecessary extra steps or interpretations.\n Step 6: Provide the final answer, starting with "FINAL ANSWER:" and using as few words as possible, simply stating the number or data point requested. \n\n The question is: {question}&lt;cot_start>Let\'s think step by step."


### DocVQA

For post-trained models, we use 0-shot config and report scores over the test set. Maximum generation length is 512 tokens. We use the following system prompt: "&lt;|image|> Read the text in the image carefully and answer the question with the text as seen exactly in the image. For yes/no questions, just respond Yes or No. If the answer is numeric, just respond with the number and nothing else. If the answer has multiple words, just respond with the words and absolutely nothing else. Never respond in a sentence or a phrase.\n Question: {question}"


### VQAv2

For post-trained models, we use 0-shot config and report scores over the test set. Maximum generation length is 25 tokens. We use the following system prompt: "&lt;|image|> Look at the image carefully and answer this visual question. For yes/no questions, just respond Yes or No. If the answer is numeric, just respond with the number and nothing else. If the answer has multiple words, just respond with the words and absolutely nothing else. Never respond in a sentence or a phrase.\n Respond with as few words as possible.\n Question: {question}"


### MathVista

For post-trained models, we use 0-shot config and report scores over the testmini set. Maximum generation length is 2048 tokens. We use an LLM-based answer extractor as recommended by MathVista paper [Lu et al. (2024)](https://arxiv.org/pdf/2310.02255). We use the following system prompt: "&lt;|image|>{question}"
