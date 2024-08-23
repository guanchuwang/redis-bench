#!/bin/bash

# Run the experiment
CUDA_VISIBLE_DEVICES=0 python rag-bench/src/rag_exp.py --k 5 --model_id mistralai/Mistral-7B-Instruct-v0.2 --retriever_name MedCPT --corpus_name RareDisease
CUDA_VISIBLE_DEVICES=1 python rag-bench/src/rag_exp.py --k 5 --model_id mistralai/Mistral-7B-Instruct-v0.2 --retriever_name MedCPT --corpus_name Textbooks
# Add more experiments as needed
# CUDA_VISIBLE_DEVICES=2 python rag-bench/src/rag_exp.py --k 5 --model_id mistralai/Mistral-7B-Instruct-v0.2 --retriever_name BM25 --corpus_name Textbooks

