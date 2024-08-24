#!/bin/bash

# Run the experiment
CUDA_VISIBLE_DEVICES=0 python rag-bench/src/rag_exp.py --k 7 --model_id mistralai/Mistral-7B-Instruct-v0.2 --retriever_name MedCPT --corpus_name RareDisease

# add more experiments as needed
# choose corpus from [RareDisease, Textbooks, StatPearls, PubMed, Wikipedia]
# choose retriever from [BM25, MedCPT]
# choose model from [mistralai/Mistral-7B-Instruct-v0.2, meta-llama/Llama-2-7b-chat-hf, google/gemma-1.1-7b-it, Qwen/Qwen2-7B-Instruct, microsoft/Phi-3-small-8k-instruct]
# choose k from [5, 7] or add more values as needed

# If you did not log in to the Hugging Face Hub, you can use add access_token to command line arguments
# Example: python rag-bench/src/rag_exp.py --k 7 --model_id mistralai/Mistral-7B-Instruct-v0.2 --retriever_name MedCPT --corpus_name RareDisease --access_token <access_token>