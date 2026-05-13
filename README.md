<<<<<<< HEAD
# LLM-HFACS

## When Planes Have Bad Days, We Figure Out Why

> *"To err is human. To analyze those errors with LLMs is... this project."*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What Is This Sorcery?

Ever wondered what *really* causes aviation incidents? Spoiler: it's rarely just one thing. It's usually a spectacular domino effect of organizational chaos, supervisory slip-ups, human factors, and that one moment where someone thought "eh, what's the worst that could happen?"

**LLM-HFACS** is a data pipeline that takes raw aviation incident reports from NASA's ASRS (Aviation Safety Reporting System) and transforms them into structured insights using the **Human Factors Analysis and Classification System (HFACS)** framework — with a little help from our AI friends.

---

## The HFACS Pyramid of "How Did We Get Here?"

```
                    ┌─────────────────────────┐
                    │   ORGANIZATIONAL        │  ← "The fish rots from the head"
                    │   INFLUENCES (L4)       │     Resource issues, climate, processes
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   UNSAFE SUPERVISION    │  ← "My boss did what now?"
                    │         (L3)            │     Inadequate oversight, ignored problems
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   PRECONDITIONS FOR     │  ← "I haven't slept in 36 hours"
                    │   UNSAFE ACTS (L2)      │     Fatigue, stress, poor communication
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │     UNSAFE ACTS (L1)    │  ← "Oops"
                    │   Errors & Violations   │     The thing that actually happened
                    └─────────────────────────┘
=======
# 🚁 LLM-HFACS

> *Teaching AI to figure out why helicopters go "oops"*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LLM-HFACS** is a research project that uses Large Language Models to automatically analyze aviation accident reports and classify them according to the **Human Factors Analysis and Classification System (HFACS)**. Because reading 215+ accident reports manually is *so* last decade.

---

## 🎯 What Does This Thing Do?

Imagine you're a safety analyst staring at a mountain of accident reports. Each one needs to be classified across 19 different human factors. Your coffee has gone cold. Your eyes are tired.

**Enter LLM-HFACS!** 🦸

This project takes those accident narratives and feeds them to LLMs that answer the eternal question: *"What went wrong and whose fault was it?"* (okay, we phrase it more scientifically than that)

```
📄 Accident Report  →  🤖 LLM Magic  →  ✅ HFACS Classification
>>>>>>> 388f7934f8c5fe9ca141aa720e8314d4be6d11e3
```

---

<<<<<<< HEAD
## Features

- **Data Pipeline**: Ingests ASRS incident data and maps narratives to HFACS categories
- **Probability Analysis**: Computes conditional probabilities across the hierarchy (L4 → L3 → L2 → L1)
- **Full Chain Analysis**: Traces complete causal paths from organizational issues to unsafe acts
- **LLM Integration**: Supports both Ollama and OpenAI models for intelligent classification
- **Subcategory Deep Dives**: Granular analysis within each HFACS level
- **Metrics & Evaluation**: Precision, recall, F1 scores — because we're data scientists, not fortune tellers

---

## Installation

```bash
# Clone the repo
git clone https://github.com/elaheoveisi/llm-hfacs.git
cd llm-hfacs

# Create a virtual environment (trust us, you want this)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Run the main analysis pipeline
python src/main.py

# Run subcategory analysis
python src/main_subcategory.py
```

---

## Project Structure

```
llm-hfacs/
├── src/
│   ├── data/           # Data loading & HFACS mapping
│   ├── features/       # Probability computations & metrics
│   ├── models/         # LLM wrappers (Ollama/OpenAI)
│   ├── visualization/  # Making numbers pretty
│   ├── main.py         # Main pipeline
│   └── main_subcategory.py
├── configs/            # YAML configurations & HFACS mappings
├── prompts/            # LLM prompt templates (CoT, ToT, IO)
├── data/
│   ├── raw/            # Raw ASRS incident data
│   └── processed/      # Output analyses
└── reports/            # Generated reports
=======
## 🧠 The HFACS Framework

HFACS is a fancy taxonomy that breaks down human errors into 4 levels:

| Level                 | What It Means                  | Examples                                                 |
| --------------------- | ------------------------------ | -------------------------------------------------------- |
| **L1: Unsafe Acts**   | The pilot did a whoopsie       | Decision errors, skill-based errors, violations          |
| **L2: Preconditions** | The stage was set for disaster | Fatigue, poor communication, bad weather                 |
| **L3: Supervision**   | The boss dropped the ball      | Inadequate supervision, planned inappropriate operations |
| **L4: Organization**  | The system is broken           | Poor resource management, toxic organizational climate   |

Think of it as a blame pyramid 🔺 — the deeper you go, the more systemic the issue!

---

## 🤖 Supported Models

We've tested this with:

| Model                  | Where It Runs           | Vibe                                  |
| ---------------------- | ----------------------- | ------------------------------------- |
| `gpt-4o-mini`          | OpenAI Cloud ☁️          | Fast and cheap, our daily driver      |
| `qwen2.5:32b-instruct` | Ollama (local/remote) 🖥️ | When you want to keep your data close |

---

## 🎪 Prompting Strategies

We don't just ask the LLM once and call it a day. We've implemented **5 different prompting strategies** to see which one makes the AI think hardest:

| Strategy             | Description                     | Complexity |
| -------------------- | ------------------------------- | ---------- |
| **IO**               | Simple yes/no questions         | ⭐          |
| **IO Expanded**      | Detailed questions with context | ⭐⭐         |
| **IO Merged**        | All factors in one mega-prompt  | ⭐⭐         |
| **Chain-of-Thought** | "Think step by step..."         | ⭐⭐⭐        |
| **Tree-of-Thought**  | Hierarchical reasoning          | ⭐⭐⭐⭐       |

---

## 📁 Project Structure

```
llm-hfacs/
├── 📂 src/
│   ├── main.py              # 🧠 The brain - runs the whole show
│   ├── models/llm.py        # 🤖 LLM wrangling (OpenAI + Ollama)
│   ├── data/                # 📊 Data loading & cleaning
│   └── features/metrics.py  # 📈 Precision, Recall, F1 - oh my!
├── 📂 prompts/
│   ├── io.yaml              # Basic prompts
│   ├── io_expanded.yaml     # Detailed prompts
│   ├── cot.yaml             # Chain-of-thought prompts
│   └── tot.yaml             # Tree-of-thought prompts
├── 📂 data/
│   ├── data.json            # Raw accident reports (the good stuff)
│   └── results/             # Where the magic outputs live
└── 📂 configs/
    └── config.yaml          # API keys and settings
>>>>>>> 388f7934f8c5fe9ca141aa720e8314d4be6d11e3
```

---

<<<<<<< HEAD
## How It Works

1. **Load** raw ASRS data (2015-2025 aviation incidents)
2. **Extract** factors from Anomaly, Human Factors, and Contributing Factors columns
3. **Map** factors to HFACS categories using predefined mappings
4. **Compute** conditional probabilities between levels
5. **Generate** full causal chains and probability matrices
6. **Profit** (in knowledge, not money — we're researchers)

---

## Tech Stack

| Category       | Tools                   |
| -------------- | ----------------------- |
| Data Wrangling | `pandas`, `openpyxl`    |
| LLM Framework  | `llama_index`           |
| LLM Providers  | Ollama, OpenAI          |
| Config         | `yaml`, `python-dotenv` |
| CLI            | `click`                 |
| Docs           | `Sphinx`                |

---

## Example Output

```
P(Inadequate_Supervision | Resource_Management) = 0.42
P(Condition_of_Operators | Inadequate_Supervision) = 0.67
P(Error | Condition_of_Operators) = 0.78

Full Chain: Resource_Management → Inadequate_Supervision → Condition_of_Operators → Error
Combined Probability: 0.22
```

*Translation: When organizations don't manage resources well, there's a 22% chance it cascades all the way down to an operational error. Fun!*

---

## Contributing

Found a bug? Have an idea? Want to add more levels to the pyramid of doom?

1. Fork it
2. Branch it (`git checkout -b feature/amazing-feature`)
3. Commit it (`git commit -m 'Add amazing feature'`)
4. Push it (`git push origin feature/amazing-feature`)
5. PR it

---

## Acknowledgments

- **NASA ASRS** for the incident data (and for making aviation safer)
- **HFACS Framework** for giving structure to chaos
- **Coffee** for making this possible

---

<p align="center">
  <i>Because every "oops" has a story, and that story has a spreadsheet.</i>
</p>
=======
## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/iHuman-Lab/llm-hfacs.git
cd llm-hfacs
pip install -r requirements.txt
```

### 2. Configure

Create/edit `configs/config.yaml` with your API keys:

```yaml
openai_api_key: "sk-your-key-here"
```

### 3. Run

```bash
python src/main.py
```

Then sit back and watch the progress bars go brrrrr 📊

---

## 📊 What You Get

After running, you'll find:

| Output                | What It Contains                 |
| --------------------- | -------------------------------- |
| `io_results.csv`      | LLM responses using IO prompting |
| `cot_results.csv`     | Chain-of-thought responses       |
| `tot_results.csv`     | Tree-of-thought responses        |
| `data/results/*.xlsx` | Precision, Recall, F1 scores     |
| Chi-squared stats     | LLM vs Human comparison          |

Plus a warm fuzzy feeling of automating tedious work ✨

---

## 🔬 Research Questions

This project investigates:

1. 🤔 Can LLMs match human experts in HFACS classification?
2. 📊 Which prompting strategy works best?
3. 🎯 How do different models compare?

**Spoiler:** Check the `data/results/` folder for answers!

---

## 📈 Metrics We Track

| Metric        | Question It Answers                            |
| ------------- | ---------------------------------------------- |
| **Precision** | When the AI says "YES", is it right?           |
| **Recall**    | Does the AI find all the factors humans found? |
| **F1 Score**  | The harmonious balance of both                 |


---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Accident       │     │    Prompting    │     │      LLM        │
│  Reports        │────▶│    Strategy     │────▶│  (GPT/Qwen)     │
│  (JSON)         │     │  (IO/CoT/ToT)   │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Results      │◀────│    Metrics      │◀────│   YES/NO        │
│   (Excel/CSV)   │     │  Calculation    │     │   Responses     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 🤝 Contributing

Found a bug? Have an idea? PRs welcome!

Just remember: we're trying to make aviation *safer*, not give the AI ideas 😅

---

## 🏢 About

Built with ❤️ at **iHuman Lab**

*Making aviation safer, one LLM query at a time.*
>>>>>>> 388f7934f8c5fe9ca141aa720e8314d4be6d11e3

<p><small>Project based on the <a target="_blank" href="https://github.com/iHuman-Lab/ihuman-cookiecutter-data-science">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

---

<p align="center">
  <i>"I asked GPT-4 to classify an accident report. It said 'skill-based error.' I felt personally attacked."</i>
  <br><br>
  🚁 Fly safe! 🚁
</p>
