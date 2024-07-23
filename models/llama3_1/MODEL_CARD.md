## Model Information

The Meta Llama 3.1 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction tuned generative models in 8B, 70B and 405B sizes (text in/text out). The Llama 3.1 instruction tuned text only models (8B, 70B, 405B) are optimized for multilingual dialogue use cases and outperform many of the available open source and closed chat models on common industry benchmarks.

**Model developer:** Meta

**Model Architecture:** Llama 3.1 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety.


<table>
  <tr>
   <td>
   </td>
   <td><strong>Training Data</strong>
   </td>
   <td><strong>Params</strong>
   </td>
   <td><strong>Input modalities</strong>
   </td>
   <td><strong>Output modalities</strong>
   </td>
   <td><strong>Context length</strong>
   </td>
   <td><strong>GQA</strong>
   </td>
   <td><strong>Token count</strong>
   </td>
   <td><strong>Knowledge cutoff</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="3" >Llama 3.1 (text only)
   </td>
   <td rowspan="3" >A new mix of publicly available online data.
   </td>
   <td>8B
   </td>
   <td>Multilingual Text
   </td>
   <td>Multilingual Text and code
   </td>
   <td>128k
   </td>
   <td>Yes
   </td>
   <td rowspan="3" >15T+
   </td>
   <td rowspan="3" >December 2023
   </td>
  </tr>
  <tr>
   <td>70B
   </td>
   <td>Multilingual Text
   </td>
   <td>Multilingual Text and code
   </td>
   <td>128k
   </td>
   <td>Yes
   </td>
  </tr>
  <tr>
   <td>405B
   </td>
   <td>Multilingual Text
   </td>
   <td>Multilingual Text and code
   </td>
   <td>128k
   </td>
   <td>Yes
   </td>
  </tr>
</table>


**Supported languages:** English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai.

**Llama 3.1 family of models**. Token counts refer to pretraining data only. All model versions use Grouped-Query Attention (GQA) for improved inference scalability.

**Model Release Date:** July 23, 2024.

**Status:** This is a static model trained on an offline dataset. Future versions of the tuned models will be released as we improve model safety with community feedback.

**License:** A custom commercial license, the Llama 3.1 Community License, is available at: [https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE)

Where to send questions or comments about the model Instructions on how to provide feedback or comments on the model can be found in the model [README](https://github.com/meta-llama/llama3). For more technical information about generation parameters and recipes for how to use Llama 3.1 in applications, please go [here](https://github.com/meta-llama/llama-recipes).


## Intended Use

**Intended Use Cases** Llama 3.1 is intended for commercial and research use in multiple languages. Instruction tuned text only models are intended for assistant-like chat, whereas pretrained models can be adapted for a variety of natural language generation tasks. The Llama 3.1 model collection also supports the ability to leverage the outputs of its models to improve other models including synthetic data generation and distillation. The Llama 3.1 Community License allows for these use cases.

**Out-of-scope** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in any other way that is prohibited by the Acceptable Use Policy and Llama 3.1 Community License. Use in languages beyond those explicitly referenced as supported in this model card.

**<span style="text-decoration:underline;">Note</span>:** Llama 3.1 has been trained on a broader collection of languages than the 8 supported languages. Developers may fine-tune Llama 3.1 models for languages beyond the 8 supported languages provided they comply with the Llama 3.1 Community License and the Acceptable Use Policy and in such cases are responsible for ensuring that any uses of Llama 3.1 in additional languages is done in a safe and responsible manner.


## Hardware and Software

**Training Factors** We used custom training libraries, Meta's custom built GPU cluster, and production infrastructure for pretraining. Fine-tuning, annotation, and evaluation were also performed on production infrastructure.

**Training Energy Use **Training utilized a cumulative of** 39.3**M GPU hours of computation on H100-80GB (TDP of 700W) type hardware, per the table below. Training time is the total GPU time required for training each model and power consumption is the peak power capacity per GPU device used, adjusted for power usage efficiency.


**Training Greenhouse Gas Emissions** Estimated total location-based greenhouse gas emissions were **11,390** tons CO2eq for training. Since 2020, Meta has maintained net zero greenhouse gas emissions in its global operations and matched 100% of its electricity use with renewable energy, therefore the total market-based greenhouse gas emissions for training were 0 tons CO2eq.


<table>
  <tr>
   <td>
   </td>
   <td><strong>Training Time (GPU hours)</strong>
   </td>
   <td><strong>Training Power Consumption (W)</strong>
   </td>
   <td><strong>Training Location-Based Greenhouse Gas Emissions</strong>
<p>
<strong>(tons CO2eq)</strong>
   </td>
   <td><strong>Training Market-Based Greenhouse Gas Emissions</strong>
<p>
<strong>(tons CO2eq)</strong>
   </td>
  </tr>
  <tr>
   <td>Llama 3.1 8B
   </td>
   <td>1.46M
   </td>
   <td>700
   </td>
   <td>420
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td>Llama 3.1 70B
   </td>
   <td>7.0M
   </td>
   <td>700
   </td>
   <td>2,040
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td>Llama 3.1 405B
   </td>
   <td>30.84M
   </td>
   <td>700
   </td>
   <td>8,930
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td>Total
   </td>
   <td>39.3M
   <td>
<ul>

</ul>
   </td>
   <td>11,390
   </td>
   <td>0
   </td>
  </tr>
</table>



The methodology used to determine training energy use and greenhouse gas emissions can be found [here](https://arxiv.org/pdf/2204.05149).  Since Meta is openly releasing these models, the training energy use and greenhouse gas emissions  will not be incurred by others.


## Training Data

**Overview:** Llama 3.1 was pretrained on ~15 trillion tokens of data from publicly available sources. The fine-tuning data includes publicly available instruction datasets, as well as over 25M synthetically generated examples.

**Data Freshness:** The pretraining data has a cutoff of December 2023.


## Benchmark scores

In this section, we report the results for Llama 3.1 models on standard automatic benchmarks. For all the evaluations, we use our internal evaluations library. Details of our evals can be found [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md). We are also releasing the raw data generated as part of our evals which can be found [here](https://huggingface.co/meta-llama).

### Base pretrained models


<table>
  <tr>
   <td><strong>Category</strong>
   </td>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong># Shots</strong>
   </td>
   <td><strong>Metric</strong>
   </td>
   <td><strong>Llama 3 8B</strong>
   </td>
   <td><strong>Llama 3.1 8B</strong>
   </td>
   <td><strong>Llama 3 70B</strong>
   </td>
   <td><strong>Llama 3.1 70B</strong>
   </td>
   <td><strong>Llama 3.1 405B</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="7" >General
   </td>
   <td>MMLU
   </td>
   <td>5
   </td>
   <td>macro_avg/acc_char
   </td>
   <td>66.7
   </td>
   <td>66.7
   </td>
   <td>79.5
   </td>
   <td>79.3
   </td>
   <td>85.2
   </td>
  </tr>
  <tr>
   <td>MMLU-Pro (CoT)
   </td>
   <td>5
   </td>
   <td>macro_avg/acc_char
   </td>
   <td>36.2
   </td>
   <td>37.1
   </td>
   <td>55.0
   </td>
   <td>53.8
   </td>
   <td>61.6
   </td>
  </tr>
  <tr>
   <td>AGIEval English
   </td>
   <td>3-5
   </td>
   <td>average/acc_char
   </td>
   <td>47.1
   </td>
   <td>47.8
   </td>
   <td>63.0
   </td>
   <td>64.6
   </td>
   <td>71.6
   </td>
  </tr>
  <tr>
   <td>CommonSenseQA
   </td>
   <td>7
   </td>
   <td>acc_char
   </td>
   <td>72.6
   </td>
   <td>75.0
   </td>
   <td>83.8
   </td>
   <td>84.1
   </td>
   <td>85.8
   </td>
  </tr>
  <tr>
   <td>Winogrande
   </td>
   <td>5
   </td>
   <td>acc_char
   </td>
   <td>-
   </td>
   <td>60.5
   </td>
   <td>-
   </td>
   <td>83.3
   </td>
   <td>86.7
   </td>
  </tr>
  <tr>
   <td>BIG-Bench Hard (CoT)
   </td>
   <td>3
   </td>
   <td>average/em
   </td>
   <td>61.1
   </td>
   <td>64.2
   </td>
   <td>81.3
   </td>
   <td>81.6
   </td>
   <td>85.9
   </td>
  </tr>
  <tr>
   <td>ARC-Challenge
   </td>
   <td>25
   </td>
   <td>acc_char
   </td>
   <td>79.4
   </td>
   <td>79.7
   </td>
   <td>93.1
   </td>
   <td>92.9
   </td>
   <td>96.1
   </td>
  </tr>
  <tr>
   <td>Knowledge reasoning
   </td>
   <td>TriviaQA-Wiki
   </td>
   <td>5
   </td>
   <td>em
   </td>
   <td>78.5
   </td>
   <td>77.6
   </td>
   <td>89.7
   </td>
   <td>89.8
   </td>
   <td>91.8
   </td>
  </tr>
  <tr>
   <td rowspan="4" >Reading comprehension
   </td>
   <td>SQuAD
   </td>
   <td>1
   </td>
   <td>em
   </td>
   <td>76.4
   </td>
   <td>77.0
   </td>
   <td>85.6
   </td>
   <td>81.8
   </td>
   <td>89.3
   </td>
  </tr>
  <tr>
   <td>QuAC (F1)
   </td>
   <td>1
   </td>
   <td>f1
   </td>
   <td>44.4
   </td>
   <td>44.9
   </td>
   <td>51.1
   </td>
   <td>51.1
   </td>
   <td>53.6
   </td>
  </tr>
  <tr>
   <td>BoolQ
   </td>
   <td>0
   </td>
   <td>acc_char
   </td>
   <td>75.7
   </td>
   <td>75.0
   </td>
   <td>79.0
   </td>
   <td>79.4
   </td>
   <td>80.0
   </td>
  </tr>
  <tr>
   <td>DROP (F1)
   </td>
   <td>3
   </td>
   <td>f1
   </td>
   <td>58.4
   </td>
   <td>59.5
   </td>
   <td>79.7
   </td>
   <td>79.6
   </td>
   <td>84.8
   </td>
  </tr>
</table>



### Instruction tuned models


<table>
  <tr>
   <td><strong>Category</strong>
   </td>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong># Shots</strong>
   </td>
   <td><strong>Metric</strong>
   </td>
   <td><strong>Llama 3 8B Instruct</strong>
   </td>
   <td><strong>Llama 3.1 8B Instruct</strong>
   </td>
   <td><strong>Llama 3 70B Instruct</strong>
   </td>
   <td><strong>Llama 3.1 70B Instruct</strong>
   </td>
   <td><strong>Llama 3.1 405B Instruct</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="4" >General
   </td>
   <td>MMLU
   </td>
   <td>5
   </td>
   <td>macro_avg/acc
   </td>
   <td>68.5
   </td>
   <td>69.4
   </td>
   <td>82.0
   </td>
   <td>83.6
   </td>
   <td>87.3
   </td>
  </tr>
  <tr>
   <td>MMLU (CoT)
   </td>
   <td>0
   </td>
   <td>macro_avg/acc
   </td>
   <td>65.3
   </td>
   <td>73.0
   </td>
   <td>80.9
   </td>
   <td>86.0
   </td>
   <td>88.6
   </td>
  </tr>
  <tr>
   <td>MMLU-Pro (CoT)
   </td>
   <td>5
   </td>
   <td>micro_avg/acc_char
   </td>
   <td>45.5
   </td>
   <td>48.3
   </td>
   <td>63.4
   </td>
   <td>66.4
   </td>
   <td>73.3
   </td>
  </tr>
  <tr>
   <td>IFEval
   </td>
   <td>
   </td>
   <td>
   </td>
   <td>76.8
   </td>
   <td>80.4
   </td>
   <td>82.9
   </td>
   <td>87.5
   </td>
   <td>88.6
   </td>
  </tr>
  <tr>
   <td rowspan="2" >Reasoning
   </td>
   <td>ARC-C
   </td>
   <td>0
   </td>
   <td>acc
   </td>
   <td>82.4
   </td>
   <td>83.4
   </td>
   <td>94.4
   </td>
   <td>94.8
   </td>
   <td>96.9
   </td>
  </tr>
  <tr>
   <td>GPQA
   </td>
   <td>0
   </td>
   <td>em
   </td>
   <td>34.6
   </td>
   <td>30.4
   </td>
   <td>39.5
   </td>
   <td>41.7
   </td>
   <td>50.7
   </td>
  </tr>
  <tr>
   <td rowspan="4" >Code
   </td>
   <td>HumanEval
   </td>
   <td>0
   </td>
   <td>pass@1
   </td>
   <td>60.4
   </td>
   <td>72.6
   </td>
   <td>81.7
   </td>
   <td>80.5
   </td>
   <td>89.0
   </td>
  </tr>
  <tr>
   <td>MBPP ++ base version
   </td>
   <td>0
   </td>
   <td>pass@1
   </td>
   <td>70.6
   </td>
   <td>72.8
   </td>
   <td>82.5
   </td>
   <td>86.0
   </td>
   <td>88.6
   </td>
  </tr>
  <tr>
   <td>Multipl-E HumanEval
   </td>
   <td>0
   </td>
   <td>pass@1
   </td>
   <td>-
   </td>
   <td>50.8
   </td>
   <td>-
   </td>
   <td>65.5
   </td>
   <td>75.2
   </td>
  </tr>
  <tr>
   <td>Multipl-E MBPP
   </td>
   <td>0
   </td>
   <td>pass@1
   </td>
   <td>-
   </td>
   <td>52.4
   </td>
   <td>-
   </td>
   <td>62.0
   </td>
   <td>65.7
   </td>
  </tr>
  <tr>
   <td rowspan="2" >Math
   </td>
   <td>GSM-8K (CoT)
   </td>
   <td>8
   </td>
   <td>em_maj1@1
   </td>
   <td>80.6
   </td>
   <td>84.5
   </td>
   <td>93.0
   </td>
   <td>95.1
   </td>
   <td>96.8
   </td>
  </tr>
  <tr>
   <td>MATH (CoT)
   </td>
   <td>0
   </td>
   <td>final_em
   </td>
   <td>29.1
   </td>
   <td>51.9
   </td>
   <td>51.0
   </td>
   <td>68.0
   </td>
   <td>73.8
   </td>
  </tr>
  <tr>
   <td rowspan="4" >Tool Use
   </td>
   <td>API-Bank
   </td>
   <td>0
   </td>
   <td>acc
   </td>
   <td>48.3
   </td>
   <td>82.6
   </td>
   <td>85.1
   </td>
   <td>90.0
   </td>
   <td>92.0
   </td>
  </tr>
  <tr>
   <td>BFCL
   </td>
   <td>0
   </td>
   <td>acc
   </td>
   <td>60.3
   </td>
   <td>76.1
   </td>
   <td>83.0
   </td>
   <td>84.8
   </td>
   <td>88.5
   </td>
  </tr>
  <tr>
   <td>Gorilla Benchmark API Bench
   </td>
   <td>0
   </td>
   <td>acc
   </td>
   <td>1.7
   </td>
   <td>8.2
   </td>
   <td>14.7
   </td>
   <td>29.7
   </td>
   <td>35.3
   </td>
  </tr>
  <tr>
   <td>Nexus (0-shot)
   </td>
   <td>0
   </td>
   <td>macro_avg/acc
   </td>
   <td>18.1
   </td>
   <td>38.5
   </td>
   <td>47.8
   </td>
   <td>56.7
   </td>
   <td>58.7
   </td>
  </tr>
  <tr>
   <td>Multilingual
   </td>
   <td>Multilingual MGSM (CoT)
   </td>
   <td>0
   </td>
   <td>em
   </td>
   <td>-
   </td>
   <td>68.9
   </td>
   <td>-
   </td>
   <td>86.9
   </td>
   <td>91.6
   </td>
  </tr>
</table>

#### Multilingual benchmarks

<table>
  <tr>
   <td><strong>Category</strong>
   </td>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Language</strong>
   </td>
   <td><strong>Llama 3.1 8B</strong>
   </td>
   <td><strong>Llama 3.1 70B</strong>
   </td>
   <td><strong>Llama 3.1 405B</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="9" ><strong>General</strong>
   </td>
   <td rowspan="9" ><strong>MMLU (5-shot, macro_avg/acc)</strong>
   </td>
   <td>Portuguese
   </td>
   <td>62.12
   </td>
   <td>80.13
   </td>
   <td>84.95
   </td>
  </tr>
  <tr>
   <td>Spanish
   </td>
   <td>62.45
   </td>
   <td>80.05
   </td>
   <td>85.08
   </td>
  </tr>
  <tr>
   <td>Italian
   </td>
   <td>61.63
   </td>
   <td>80.4
   </td>
   <td>85.04
   </td>
  </tr>
  <tr>
   <td>German
   </td>
   <td>60.59
   </td>
   <td>79.27
   </td>
   <td>84.36
   </td>
  </tr>
  <tr>
   <td>French
   </td>
   <td>62.34
   </td>
   <td>79.82
   </td>
   <td>84.66
   </td>
  </tr>
  <tr>
   <td>Hindi
   </td>
   <td>50.88
   </td>
   <td>74.52
   </td>
   <td>80.31
   </td>
  </tr>
  <tr>
   <td>Thai
   </td>
   <td>50.32
   </td>
   <td>72.95
   </td>
   <td>78.21
   </td>
  </tr>
</table>



## Responsibility & Safety

As part of our Responsible release approach, we followed a three-pronged strategy to managing trust & safety risks:



* Enable developers to deploy helpful, safe and flexible experiences for their target audience and for the use cases supported by Llama.
* Protect developers against adversarial users aiming to exploit Llama capabilities to potentially cause harm.
* Provide protections for the community to help prevent the misuse of our models.


### Responsible deployment

Llama is a foundational technology designed to be used in a variety of use cases, examples on how Meta’s Llama models have been responsibly deployed can be found in our [Community Stories webpage](https://llama.meta.com/community-stories/). Our approach is to build the most helpful models enabling the world to benefit from the technology power, by aligning our model safety for the generic use cases addressing a standard set of harms. Developers are then in the driver seat to tailor safety for their use case, defining their own policy and deploying the models with the necessary safeguards in their Llama systems. Llama 3.1 was developed following the best practices outlined in our Responsible Use Guide, you can refer to the [Responsible Use Guide](https://llama.meta.com/responsible-use-guide/) to learn more.


#### Llama 3.1 instruct

Our main objectives for conducting safety fine-tuning are to provide the research community with a valuable resource for studying the robustness of safety fine-tuning, as well as to offer developers a readily available, safe, and powerful model for various applications to reduce the developer workload to deploy safe AI systems. For more details on the safety mitigations implemented please read the Llama 3 paper.

**Fine-tuning data**

We employ a multi-faceted approach to data collection, combining human-generated data from our vendors with synthetic data to mitigate potential safety risks. We’ve developed many large language model (LLM)-based classifiers that enable us to thoughtfully select high-quality prompts and responses, enhancing data quality control.

**Refusals and Tone**

Building on the work we started with Llama 3, we put a great emphasis on model refusals to benign prompts as well as refusal tone. We included both borderline and adversarial prompts in our safety data strategy, and modified our safety data responses to follow  tone guidelines.


#### Llama 3.1 systems

**Large language models, including Llama 3.1, are not designed to be deployed in isolation but instead should be deployed as part of an overall AI system with additional safety guardrails as required.** Developers are expected to deploy system safeguards when building agentic systems. Safeguards are key to achieve the right helpfulness-safety alignment as well as mitigating safety and security risks inherent to the system and any integration of the model or system with external tools.

As part of our responsible release approach, we provide the community with [safeguards](https://llama.meta.com/trust-and-safety/) that developers should deploy with Llama models or other LLMs, including Llama Guard 3, Prompt Guard and Code Shield. All our [reference implementations](https://github.com/meta-llama/llama-agentic-system) demos contain these safeguards by default so developers can benefit from system-level safety out-of-the-box.


#### New capabilities

Note that this release introduces new capabilities, including a longer context window, multilingual inputs and outputs and possible integrations by developers with third party tools. Building with these new capabilities requires specific considerations in addition to the best practices that generally apply across all Generative AI use cases.

**Tool-use**: Just like in standard software development, developers are responsible for the integration of the LLM with the tools and services of their choice. They should define a clear policy for their use case and assess the integrity of the third party services they use to be aware of the safety and security limitations when using this capability. Refer to the Responsible Use Guide for best practices on the safe deployment of the third party safeguards.

**Multilinguality**: Llama 3.1 supports 7 languages in addition to English: French, German, Hindi, Italian, Portuguese, Spanish, and Thai. Llama may be able to output text in other languages than those that meet performance thresholds for safety and helpfulness. We strongly discourage developers from using this model to converse in non-supported languages without implementing finetuning and system controls in alignment with their policies and the best practices shared in the Responsible Use Guide.


### Evaluations

We evaluated Llama models for common use cases as well as specific capabilities. Common use cases evaluations measure safety risks of systems for most commonly built applications including chat bot, coding assistant, tool calls. We built dedicated, adversarial evaluation datasets and evaluated systems composed of Llama models and Llama Guard 3 to filter input prompt and output response. It is important to evaluate applications in context, and we recommend building dedicated evaluation dataset for your use case. Prompt Guard and Code Shield are also available if relevant to the application.

Capability evaluations measure vulnerabilities of Llama models inherent to specific capabilities, for which were crafted dedicated benchmarks including long context, multilingual, tools calls, coding or memorization.

**Red teaming**

For both scenarios, we conducted recurring red teaming exercises with the goal of discovering risks via adversarial prompting and we used the learnings to improve our benchmarks and safety tuning datasets.

We partnered early with subject-matter experts in critical risk areas to understand the nature of these real-world harms and how such models may lead to unintended harm for society. Based on these conversations, we derived a set of adversarial goals for the red team to attempt to achieve, such as extracting harmful information or reprogramming the model to act in a potentially harmful capacity.  The red team consisted of experts in cybersecurity, adversarial machine learning, responsible AI, and integrity in addition to multilingual content specialists with background in integrity issues in specific geographic markets. .


### Critical and other risks

We specifically focused our efforts on mitigating the following critical risk areas:

**1- CBRNE (Chemical, Biological, Radiological, Nuclear, and Explosive materials) helpfulness**

To assess risks related to proliferation of chemical and biological weapons, we performed uplift testing designed to assess whether use of Llama 3.1 models could meaningfully increase the capabilities of malicious actors to plan or carry out attacks using these types of weapons.


**2. Child Safety**

Child Safety risk assessments were conducted using a team of experts, to assess the model’s capability to produce outputs that could result in Child Safety risks and inform on any necessary and appropriate risk mitigations via fine tuning. We leveraged those expert red teaming sessions to expand the coverage of our evaluation benchmarks through Llama 3 model development.  For Llama 3, we conducted new in-depth sessions using objective based methodologies to assess the model risks along multiple attack vectors including the additional languages Llama 3 is trained on. We also partnered with content specialists to perform red teaming exercises assessing potentially violating content while taking account of market specific nuances or experiences.

**3. Cyber attack enablement**

Our cyber attack uplift study investigated whether LLMs can enhance human capabilities in hacking tasks, both in terms of skill level and speed.

Our attack automation study focused on evaluating the capabilities of LLMs when used as autonomous agents in cyber offensive operations, specifically in the context of ransomware attacks. This evaluation was distinct from previous studies that considered LLMs as interactive assistants. The primary objective was to assess whether these models could effectively function as independent agents in executing complex cyber-attacks without human intervention.

Our study of Llama-3.1-405B’s social engineering uplift for cyber attackers was conducted to assess the effectiveness of AI models in aiding cyber threat actors in spear phishing campaigns. Please read our Llama 3.1 Cyber security whitepaper to learn more.


### Community

Generative AI safety requires expertise and tooling, and we believe in the strength of the open community to accelerate its progress. We are active members of open consortiums, including the AI Alliance, Partnership on AI and MLCommons, actively contributing to safety standardization and transparency. We encourage the community to adopt taxonomies like the MLCommons Proof of Concept evaluation to facilitate collaboration and transparency on safety and content evaluations. Our Purple Llama tools are open sourced for the community to use and widely distributed across ecosystem partners including cloud service providers. We encourage community contributions to our [Github repository](https://github.com/meta-llama/PurpleLlama).

We also set up the [Llama Impact Grants](https://llama.meta.com/llama-impact-grants/) program to identify and support the most compelling applications of Meta’s Llama model for societal benefit across three categories: education, climate and open innovation. The 20 finalists from the hundreds of applications can be found [here](https://llama.meta.com/llama-impact-grants/#finalists).

Finally, we put in place a set of resources including an [output reporting mechanism](https://developers.facebook.com/llama_output_feedback) and [bug bounty program](https://www.facebook.com/whitehat) to continuously improve the Llama technology with the help of the community.


## Ethical Considerations and Limitations

The core values of Llama 3.1 are openness, inclusivity and helpfulness. It is meant to serve everyone, and to work for a wide range of use cases. It is thus designed to be accessible to people across many different backgrounds, experiences and perspectives. Llama 3.1 addresses users and their needs as they are, without insertion unnecessary judgment or normativity, while reflecting the understanding that even content that may appear problematic in some cases can serve valuable purposes in others. It respects the dignity and autonomy of all users, especially in terms of the values of free thought and expression that power innovation and progress.

But Llama 3.1 is a new technology, and like any new technology, there are risks associated with its use. Testing conducted to date has not covered, nor could it cover, all scenarios. For these reasons, as with all LLMs, Llama 3.1’s potential outputs cannot be predicted in advance, and the model may in some instances produce inaccurate, biased or other objectionable responses to user prompts. Therefore, before deploying any applications of Llama 3.1 models, developers should perform safety testing and tuning tailored to their specific applications of the model. Please refer to available resources including our [Responsible Use Guide](https://llama.meta.com/responsible-use-guide), [Trust and Safety](https://llama.meta.com/trust-and-safety/) solutions, and other [resources](https://llama.meta.com/docs/get-started/) to learn more about responsible development.
