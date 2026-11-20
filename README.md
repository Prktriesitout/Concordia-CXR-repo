```markdown
# 🩻 Multi-Model Chest X-Ray Analysis & Consensus System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/) 
[![Gradio UI](https://img.shields.io/badge/UI-Gradio-teal.svg)](https://gradio.app/)
[![Hardware](https://img.shields.io/badge/GPU-Colab%20T4%20|%20L4%20|%20A100-orange.svg)](https://colab.research.google.com/)
[![Model](https://img.shields.io/badge/Local%20Model-MedGemma%201.5--4B--IT-red.svg)](https://huggingface.co/google/medgemma-1.5-4b-it)
[![API](https://img.shields.io/badge/API-GPT--6%20Astra-blueviolet.svg)](https://openai.com/)

An advanced, multi-agent AI framework designed to simulate independent radiology opinions and compile a consensus clinical research report from Chest X-rays. Combining a specialized local medical VLM (**Google MedGemma**) with a cutting-edge proprietary API (**OpenAI GPT-6 Astra**), the pipeline validates observations, flags clinical conflicts, and writes a unified final report.

---

## 📌 System Architecture & Pipeline Flow

```
                    +------------------------------+
                    |    Input Chest X-Ray (Img)   |
                    +--------------+---------------+
                                   |
         +-------------------------+-------------------------+
         |                                                   |
         v (Local GPU Run)                                   v (Cloud API Run)
+-------------------------------+                 +-------------------------------+
|    Model 1: MedGemma 4B       |                 |     Model 2: GPT-6 Astra      |
|  4-bit Quantized / FP32 Comp  |                 |     Original Image Detail     |
+---------------+---------------+                 +---------------+---------------+
                |                                                 |
                | [Draft Report 1]               [Draft Report 2] |
                +------------------------+------------------------+
                                         | 
                                         v
                        +---------------------------------+
                        |  Model 3: GPT-6 Consensus Agent |
                        |   - Image Evidence Re-Review    |
                        |   - Conflict Resolution         |
                        |   - Clinical Synthesizer        |
                        +----------------+----------------+
                                         |
                                         v
                        +---------------------------------+
                        |   Consolidated PDF/MD Report    |
                        |   Gradio App Streamed Output    |
                        +---------------------------------+
```

---

## ✨ Key Features

* **🧠 Hybrid Multi-Agent Model Ensemble**:
  * **Model 1 (Local Local-VLM)**: Runs **Google MedGemma 1.5 4B IT** directly on a single Colab T4 GPU (using 4-bit quantization with FP32 activation compute to avoid numeric overflow / empty reports).
  * **Model 2 (API Model)**: Queries **OpenAI GPT-6 Astra** directly to obtain a fully independent, highly detailed visual analysis.
  * **Model 3 (Consensus Adjudication)**: A second-pass verification agent (GPT-6 Astra Reviewer) that compares both reports against the original image, highlighting agreements, resolving differences, and finalizing clinical impressions.
* **⚡ Intelligent Double-Layer Caching**:
  * **In-session cache**: Skips loading and quantization overhead entirely when executing repeatedly in active memory.
  * **Persistent cache**: Automated backup and recovery to/from Google Drive to instantly restore quantized weights in seconds on fresh runtimes.
* **🌐 Live Streaming Gradio Web UI**:
  * Parallel pipeline tracking with clean, state-based UI status cards.
  * Real-time streaming output of the consensus review.
  * Multi-file download hub to fetch separate drafts as well as the consolidated comprehensive markdown package.

---

## 🛠️ Installation & Setup

### Prerequisite Secrets in Colab
Set up the following keys inside your Colab Secrets (🔑 icon on the left panel):
1. `OpenAI`: Your OpenAI platform API key.
2. `HF_TOKEN`: *(Optional)* Your Hugging Face user access token (required if accessing gated models; make sure you accepted the MedGemma license on its HF model card page first).

### Manual Installation
```bash
pip install -q -U "transformers>=4.50.0" accelerate bitsandbytes openai pillow "gradio>=5"
```

---

## 🖥️ Launching the Interactive Interface

Simply run the main code cell inside your notebook to deploy the interactive Gradio dashboard. 

```python
# Execution command
demo.launch(share=True)
```

This generates a public, shareable link enabling access on mobile devices or external web-browsers.

---

## 📁 Directory Structure of Generated Reports

Each analysis execution packages and saves the clinical logs directly to your runtime workspace:

```
/content/Xray_Analysis_Reports/
└── [Timestamp_Folder]/
    ├── 00_Full_Analysis.md         <-- Complete merged analysis package
    ├── 01_MedGemma_Report.md       <-- Model 1 raw independent opinion 
    ├── 02_GPT6_Astra_Report.md     <-- Model 2 raw independent opinion
    └── 03_Final_Verified_Report.md <-- Model 3 reviewed final consensus report
```

---

## ⚠️ Clinical Research Disclaimer

> **IMPORTANT:** This repository and the tools within are constructed solely for **artificial intelligence research, demonstration, and educational validation**. It is **not** cleared, certified, or intended for medical diagnostic decisions or clinical environments. All generated reports must be evaluated and verified by a board-certified clinical radiologist before any medical decisions or patient management paths are planned.
```
