#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python rag_experiment.py --k 5 --llm_name mistralai/Mistral-7B-Instruct-v0.2 --retriever_name MedCPT --corpus_name Textbooks

# add more experiments as wanted