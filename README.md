# Assessing and Enhancing Large Language Models in Rare Disease Question-answering

This is the official codebase of paper _Assessing and Enhancing Large Language Models in Rare Disease Question-answering_.

## Dataset Overview

ReDis-QA dataset widely covers 205 types of rare diseases, where the most frequent disease features over 100 questions.

<img width="900" height="260" src="https://github.com/guanchuwang/redis-bench/blob/main/figures/disease_freq.png">

ReDis-QA dataset includes 11\%, 33\%, 13\%, 15\%, 18\% of the questions corresponding to the symptoms, causes, affects, related-disorders, diagnosis of rare diseases, respectively. 
The remaining 9\% of the questions pertain to other properties of the diseases.

<img width="400" height="290" src="https://github.com/guanchuwang/redis-bench/blob/main/figures/theme_ratio.png">


## Dependency
```
numpy
scikit-learn
scipy
torch
accelerate==0.32.1
transformers==4.42.4
datasets==2.20.0
ipdb
tqdm
```

## Quick Exploration on the Benchmark

Run LLMs w/o RAG on the ResDis-QA dataset:
```bash
cd zero-shot-bench
bash ./scripts/run_exp.sh
```

The accuracy of LLMs on each subset of properties is shown as follows:

<img width="600" height="290" src="https://github.com/guanchuwang/redis-bench/blob/main/figures/llm_results.png">

Run RAG with ReCOP corpus on the ResDis-QA dataset:
```bash
cd meta-data-bench
bash ./scripts/run_exp.sh
```

The accuracy of RAG with ReCOP corpus is shown as follows:

<img width="200" height="200" src="./figures/radar_Mistral-7B-v0.2.png">
<img width="200" height="200" src="./figures/radar_Gemma-1.1-7B.png">
<img width="200" height="200" src="./figures/radar_Phi-3-7B.png">
<img width="200" height="200" src="./figures/radar_Qwen-2-7B.png">


