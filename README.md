# 🔐 CyberSec-Reasoner 

> **A Three-Stage Supervised Reasoning + GRPO Fine-Tuned LLM for Cybersecurity Vulnerability Analysis**
A 3-stage Post-Trained Reasoning LLM 

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/sajan-sarker/CyberSec-Reasoner/blob/main/LICENSE)
[![Model: Qwen3.5-4B](https://img.shields.io/badge/Base%20Model-Qwen3.5--4B-orange)](https://huggingface.co/Qwen/Qwen3.5-4B)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-sajan--sarker-yellow)](https://huggingface.co/sajan-sarker/Qwen3.5-4B-LoRA-GRPO-CyberSec-Reasoner)
[![Framework: TRL](https://img.shields.io/badge/Framework-TRL%20%2F%20Transformers-green)](https://github.com/huggingface/trl)

---

## 📖 Overview

**CyberSec-Reasoner** is a domain-specialized Large Language Model (LLM) engineered for structured cybersecurity reasoning. Built on top of `Qwen/Qwen3.5-4B`, it undergoes a rigorous three-stage post-training pipeline — combining Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) — to produce a model capable of explainable vulnerability analysis.

The model doesn't just predict labels; it **reasons** its way to answers using explicit `<think>...</think>` Chain-of-Thought (CoT) traces, making it interpretable for security analysts.

### 🎯 Core Capabilities

| Capability | Description |
|---|---|
| 🔍 **CVE → CWE Mapping** | Classifies CVE descriptions to appropriate CWE identifiers |
| 🧩 **Attack Scenario Decomposition** | Breaks down complex intrusion narratives into structured analyses |
| 🛡️ **Structured Reasoning** | Generates `<think>...</think>` traces for transparent decision-making |
| 📚 **Explainable AI** | Produces interpretable outputs suitable for security analyst review |

---

## 🏗️ Model Architecture

| Component | Specification |
|---|---|
| **Base Model** | `Qwen/Qwen3.5-4B` by Alibaba |
| **Tuning Method** | QLoRA (8-bit) + LoRA (16-bit) |
| **PEFT Strategy** | Hugging Face PEFT library |
| **RL Algorithm** | Group Relative Policy Optimization (GRPO) |
| **Model Size** | ~9 GB |
| **Hardware Requirement** | GPU ≥ 16GB VRAM recommended |
| **Precision** | `bfloat16` |

---

## 🤗 Model on HuggingFace

The fully merged model (no adapter loading required) is publicly available on HuggingFace:

**👉 [sajan-sarker/Qwen3.5-4B-LoRA-GRPO-CyberSec-Reasoner](https://huggingface.co/sajan-sarker/Qwen3.5-4B-LoRA-GRPO-CyberSec-Reasoner)**

```python
model_name = "sajan-sarker/Qwen3.5-4B-LoRA-GRPO-CyberSec-Reasoner"
```

---

## 🔬 Three-Stage Post-Training Pipeline
![alt text](https://github.com/sajan-sarker/CyberSec-Reasoner/blob/main/docs/Training%20Workflow%20Diagram.png?raw=true)

### Stage 1 — Knowledge-Based SFT
- **Goal:** Teach foundational cybersecurity vocabulary and domain facts
- **Data:** Merged dataset from [Fenrir v2.0](https://huggingface.co/datasets/AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.0) + [Cybersec-Reasoning-Merged](https://huggingface.co/datasets/Mohannadcse/cybersec-reasoning-merged)
- **Method:** SFT with full loss on prompt + assistant tokens (~96K samples, 3 epochs)

### Stage 2 — Cold-Start Reasoning SFT
- **Goal:** Establish the structured `<think>...</think>` reasoning format
- **Data:** ~1,000 samples from [Primus-Reasoning](https://huggingface.co/datasets/trendmicro-ailab/Primus-Reasoning), filtered for CVE-to-CWE mapping
- **Method:** SFT with **assistant-only loss** to enforce the reasoning structure

### Stage 3 — GRPO Reinforcement Learning
- **Goal:** Refine judgment and improve CWE prediction accuracy
- **Data:** ~2,300 CVE-to-CWE samples from the Primus dataset
- **Method:** GRPO with a composite reward function (see below)

---

## 🏆 Reward System (GRPO Stage)

The GRPO stage uses a **Weighted Hybrid Reward Function**:

$$\text{Total Reward} = 0.5 \cdot R_{\text{cwe}} + 0.3 \cdot R_{\text{format}} + 0.2 \cdot R_{\text{keyword}}$$

| Reward Component | Weight | Description |
|---|---|---|
| **CWE Accuracy** ($R_{\text{cwe}}$) | 0.5 | 1 if predicted CWE-ID matches ground truth, else 0 |
| **Format Integrity** ($R_{\text{format}}$) | 0.3 | Rewards correct use of `<think>...</think>` block |
| **Semantic Grounding** ($R_{\text{keyword}}$) | 0.2 | Rewards inclusion of relevant security keywords in reasoning |

---

## 📊 Evaluation Results

### GRPO Phase — Final Model Evaluation
> Evaluated on 1,000 samples from [cve-and-cwe-dataset-1999-2025](https://huggingface.co/datasets/stasvinokur/cve-and-cwe-dataset-1999-2025)

| Metric | Score |
|---|---|
| **CWE Accuracy (Exact Match)** | 33.90% |
| **Format Match** (`<think>` structure) | 94.85% |
| **Keyword Alignment Accuracy** | 97.85% |

> **Note:** The relatively lower CWE exact-match accuracy is expected given the 4B parameter scale and limited CVE–CWE knowledge coverage. The model is optimized for **structured reasoning quality**, and when augmented with a RAG pipeline or CVE–CWE database, is expected to achieve significantly higher mapping accuracy.

---

### SFT Phase — Language Modeling Metrics
> Evaluated on the test split of the SFT stage-1 dataset

| Metric | Score |
|---|---|
| **ROUGE-1** | 0.3680 |
| **ROUGE-2** | 0.1528 |
| **ROUGE-L** | 0.1779 |
| **ROUGE-Lsum** | 0.2507 |
| **BLEU** | 0.1098 |
| **Perplexity** | 1.7989 |

> A perplexity of ~1.80 indicates the model has developed strong domain fluency and predictive confidence over cybersecurity text after Stage 1 SFT.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install --upgrade transformers==5.5.4
```

### Option 1 — Quick Start with Wrapper Class (Recommended)

```python
# Download the inference wrapper from the HuggingFace repo
import subprocess
subprocess.run(["wget", "https://huggingface.co/sajan-sarker/Qwen3.5-4B-LoRA-GRPO-CyberSec-Reasoner/resolve/main/inference.py"])

from inference import CyberSecReasoner, CyberSecReasonerConfig

cfg = CyberSecReasonerConfig()
model = CyberSecReasoner(cfg)

cve_description = """Analyze the following CVE description and map it to the appropriate CWE.
Provide a brief justification for your choice. Ensure the last line of your response contains only the CWE ID.
CVE Description: Cross-site scripting (XSS) vulnerability in MyBulletinBoard (MyBB) allows remote
attackers to inject arbitrary web script or HTML via a signature containing a JavaScript URI in the
SRC attribute of an IMG element..."""

messages = [{"role": "user", "content": cve_description.strip()}]

completion = model.generate(messages)
reasoning, final_output = model.format_completions(completion)

print("=== Reasoning Trace ===")
print(reasoning)

print("\n=== Analysis & CWE Classification ===")
print(final_output)
```

### Option 2 — Direct Transformers Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "sajan-sarker/Qwen3.5-4B-LoRA-GRPO-CyberSec-Reasoner"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

cve_description = """Analyze the following CVE description and map it to the appropriate CWE.
Provide a brief justification for your choice. Ensure the last line of your response contains only the CWE ID.
CVE Description: <YOUR CVE DESCRIPTION HERE>"""

messages = [{"role": "user", "content": cve_description.strip()}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
print(tokenizer.decode(generated_tokens, skip_special_tokens=True))
```

### Expected Output Format

```
<think>
The CVE describes a Cross-Site Scripting vulnerability where user-controlled input
(a JavaScript URI in an IMG SRC attribute) is embedded into HTML output without
proper sanitization. The key indicator is the use of SGML numeric character references
to bypass filters...
</think>

The CVE describes an XSS vulnerability caused by improper neutralization of
user-controllable input embedded into HTML output. This directly aligns with
the CWE category for improper neutralization of input during web page generation.

CWE-79
```

---

## 📁 Data Pipeline

### Input Format

All training data is standardized into the following message schema:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a cybersecurity expert. Analyze vulnerabilities and map them to CWE identifiers."
    },
    {
      "role": "user",
      "content": "Analyze the following CVE description and map it to the appropriate CWE..."
    },
    {
      "role": "assistant",
      "content": "<think>\n[Reasoning trace]\n</think>\n\n[Technical explanation]\n\nCWE-XXX"
    }
  ]
}
```

### Datasets Used

| Stage | Dataset | Size | Source |
|---|---|---|---|
| SFT Stage 1 | Cybersecurity-Dataset-Fenrir-v2.0 | ~80K | HuggingFace |
| SFT Stage 1 | Cybersec-Reasoning-Merged | ~16K | HuggingFace |
| SFT Stage 2 | Primus-Reasoning (CVE-CWE filtered) | ~1K | TrendMicro AI Lab |
| GRPO Stage 3 | Primus-Reasoning (CVE-CWE filtered) | ~2.3K | TrendMicro AI Lab |
| Evaluation | cve-and-cwe-dataset-1999-2025 | 1,000 | HuggingFace |

---

## 🛠️ Tech Stack

| Category | Tool / Library | Version |
|---|---|---|
| **Base Model** | `Qwen/Qwen3.5-4B` | — |
| **Training Framework** | `transformers` | ≥ 5.5.4 |
| **RL Framework** | `trl` (Transformer Reinforcement Learning) | 1.0.0 |
| **PEFT** | `peft` (Hugging Face) | 0.18.1 |
| **Deep Learning** | `torch` | 2.11.0 |
| **Acceleration** | `accelerate` | 1.13.0 |
| **Quantization** | `bitsandbytes` (8-bit QLoRA) | 0.49.2 |
| **Evaluation** | `rouge-score`, `sacrebleu`, `nltk` | latest |
| **Hardware** | AMD MI300X (192GB GPU) | — |
| **Cloud** | AMD Developer Cloud — ATL1 | — |

---

## 📋 Training Hyperparameters

| Hyperparameter | Stage 1 (SFT) | Stage 2 (Cold-Start) | Stage 3 (GRPO) |
|---|---|---|---|
| **Method** | QLoRA | LoRA | QLoRA |
| **Precision** | 8-bit | 16-bit | 8-bit |
| **Epochs** | 3 | 2 | 1 |
| **Dataset Size** | ~96K samples | ~1K samples | ~2.3K samples |
| **Loss Type** | Full (prompt + response) | Assistant only | GRPO reward |

---

## 🗂️ Repository Structure

```
CyberSec-Reasoner/
│
├── docs/                      # Necessary documents
├── evaluation/                # Model evaluation scripts
├── models/                    # Trained models and Adapters
├── notebooks/                 # Jupyter notebooks for exploration
├── outputs/                   # Generated outputs during evaluations
├── plots/                     # Saved plots during training and after evaluations
├── src/                       # All the source code
├── training/                  # All the training scripts
├── wandb/                     # WandB logs
├── requirements.txt
└── README.md
```

---

## ⚠️ Limitations & Recommendations

### Known Limitations

- Limited coverage of the full CVE–CWE mapping space
- May produce incorrect or hallucinated CWE IDs for uncommon vulnerability types
- Performance depends on reasoning quality rather than memorization
- Not all security domains are equally represented in training data
- Small 4B parameter scale limits breadth of encoded knowledge

### Recommendations for Use

- ✅ Always verify outputs against authoritative sources (NVD, MITRE CWE database)
- ✅ Combine with a **RAG pipeline** backed by a CVE–CWE database for improved accuracy
- ✅ Use for analyst-assist workflows, not fully automated decision-making
- ✅ Pair with human expert review in high-stakes environments
- ❌ Do not use for real-time production defense systems without validation
- ❌ Do not use for generating exploit strategies or malicious purposes

---

## 🔭 Future Work

- [ ] Expand training data coverage for rare CWE categories
- [ ] Integrate RAG pipeline with live NVD/MITRE database
- [ ] Scale to larger base model (7B / 14B) for improved knowledge breadth
- [ ] Add MITRE ATT&CK® tactic/technique mapping capabilities
- [ ] Build a web UI for analyst-friendly interaction
- [ ] Multi-hop reasoning over chained CVE records

---

## 📄 License

This project is licensed under the **Apache License 2.0**, inherited from the base model `Qwen/Qwen3.5-4B` by Alibaba. See the [LICENSE](LICENSE) file for full terms.

---

## 👤 Author

**Sajan Kumer Sarker**

For questions, feedback, or collaboration opportunities, please open an issue on the [HuggingFace repository](https://huggingface.co/sajan-sarker/Qwen3.5-4B-LoRA-GRPO-CyberSec-Reasoner) or this GitHub repo.

---

## 📚 Citation

If you use this model or methodology in your research, please cite:

```bibtex
@misc{sarker2025cybersec,
  author       = {Sajan Kumer Sarker},
  title        = {CyberSec-Reasoner: A Three-Stage Supervised Reasoning + GRPO Fine-Tuned LLM for Cybersecurity Tasks},
  year         = {2025},
  publisher    = {HuggingFace},
  howpublished = {\url{https://huggingface.co/sajan-sarker/Qwen3.5-4B-LoRA-GRPO-CyberSec-Reasoner}}
}
```

---

## 🙏 Acknowledgements

- [Alibaba Qwen Team](https://huggingface.co/Qwen) for the Qwen3.5-4B base model
- [Hugging Face TRL Team](https://github.com/huggingface/trl) for the GRPO training framework
- [TrendMicro AI Lab](https://huggingface.co/trendmicro-ailab) for the Primus-Reasoning dataset
- [AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud.html) for compute resources
