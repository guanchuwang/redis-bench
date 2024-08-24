# <img width="30" height="30" src="./figures/logo.png"> Assessing and Enhancing Large Language Models in Rare Disease Question-answering


This is the official codebase of paper _Assessing and Enhancing Large Language Models in Rare Disease Question-answering_.


## Resources
:star2: Please star our repo to follow the latest updates on <img width="15" height="15" src="./figures/logo.png"> ReDis-QA-Bench!

:mega: We have released our [paper](https://arxiv.org/pdf/2408.08422) and source code of ReDis-QA-Bench!

:orange_book: We have released our benchmark dataset [ReDis-QA](https://huggingface.co/datasets/guan-wang/ReDis-QA)!

:closed_book: We have released our corpus for RAG [ReCOP](https://huggingface.co/datasets/guan-wang/ReCOP)!

:blue_book: Baseline corpus refers to [PubMed](https://huggingface.co/datasets/MedRAG/pubmed), [Textbook](https://huggingface.co/datasets/MedRAG/textbooks), [Wikipedia](https://huggingface.co/datasets/MedRAG/wikipedia) and [StatPearls](https://huggingface.co/datasets/MedRAG/statpearls)!



## Dataset Overview

ReDis-QA dataset widely covers 205 types of rare diseases, where the most frequent disease features over 100 questions.

<img width="900" height="260" src="./figures/disease_freq.png">

ReDis-QA dataset includes 11\%, 33\%, 13\%, 15\%, 18\% of the questions corresponding to the symptoms, causes, affects, related-disorders, diagnosis of rare diseases, respectively. 
The remaining 9\% of the questions pertain to other properties of the diseases.

<img width="400" height="290" src="./figures/theme_ratio.png">


## Requirements

1. **Python Environment**:  
   - Create a virtual environment using Python 3.10.0.

2. **PyTorch Installation**:  
   - Install the version of PyTorch that is compatible with your system's CUDA version (e.g., PyTorch 2.4.0+cu121).

3. **Additional Libraries**:  
   - Install the remaining required libraries by running:
     ```bash
     pip install -r requirements.txt
     ```

4. **Git Large File Storage (Git LFS)**:  
   - Git LFS is required to download and load large corpora Textbooks, Wikipedia, and PubMed for the first time. ReCOP downloading does not require Git LFS.

5. **Java**:  
   - Ensure Java is installed for using the BM25 retriever.


## Quick Exploration on the Benchmark

Loading ReDis-QA Dataset:
```bash
from datasets import load_dataset
eval_dataset = load_dataset("guan-wang/ReDis-QA")['test'] 
```
Loading ReCOP Corpus:
```bash
from datasets import load_dataset
corpus = load_dataset("guan-wang/ReCOP")['train'] 
```

Run LLMs w/o RAG on the ResDis-QA dataset:
```bash
bash zero-shot-bench/scripts/run_exp.sh
```

The accuracy of LLMs on each subset of properties is shown as follows:

<img width="600" height="290" src="https://github.com/guanchuwang/redis-bench/blob/main/figures/llm_results.png">

Run RAG with ReCOP corpus using the meta-data retriever on the ResDis-QA dataset:
```bash
bash meta-data-bench/scripts/run_exp.sh
```

Run RAG with ReCOP and baseline corpora using MedCPT/BM25 retriever on the ResDis-QA dataset:
```bash
bash rag-bench/scripts/run_exp.sh
```

The accuracy of RAG with ReCOP corpus is shown as follows:

<img width="200" height="200" src="./figures/radar_Mistral-7B-v0.2.png">&nbsp;<img width="200" height="200" src="./figures/radar_Gemma-1.1-7B.png">&nbsp;<img width="200" height="200" src="./figures/radar_Phi-3-7B.png">&nbsp;<img width="200" height="200" src="./figures/radar_Qwen-2-7B.png">

Run RAG with baseline corpus and combine with ReCOP on the ResDis-QA dataset:
```bash
bash combine-corpora-bench/scripts/run_exp.sh
```

<img width="200" height="200" src="./figures/radar_PubMed.png">&nbsp;<img width="200" height="200" src="./figures/radar_Textbooks.png">&nbsp;<img width="200" height="200" src="./figures/radar_Wikipedia.png">&nbsp;<img width="200" height="200" src="./figures/radar_StatPearls.png">

## Acknowledgement

The MedCPT, BM25 retrievers, and baseline corpus are sourced from the opensource repo [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG). 
Thanks to their contributions to the community!

## Cite This Work

If you find this work useful, you may cite this work:

````
@article{wang2024assessing,
  title={Assessing and Enhancing Large Language Models in Rare Disease Question-answering},
  author={Wang, Guanchu and Ran, Junhao and Tang, Ruixiang and Chang, Chia-Yuan and Chuang, Yu-Neng and Liu, Zirui and Braverman, Vladimir and Liu, Zhandong and Hu, Xia},
  journal={arXiv preprint arXiv:2408.08422},
  year={2024}
}
````
