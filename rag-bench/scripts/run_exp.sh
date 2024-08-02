#!/bin/bash

# Run the experiment
CUDA_VISIBLE_DEVICES=0 python rag-bench/src/rag_exp.py --k 5 --llm_name mistralai/Mistral-7B-Instruct-v0.2 --retriever_name MedCPT --corpus_name Textbooks

# Add more experiments as needed
# python rag_experiment.py --k 5 --llm_name mistralai/Mistral-7B-Instruct-v0.2 --retriever_name BM25 --corpus_name Textbooks

