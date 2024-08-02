import sys
from src.medrag import MedRAG
import os
import torch

# Set the CUDA_VISIBLE_DEVICES environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

medrag = MedRAG(llm_name="mistralai/Mistral-7B-Instruct-v0.2", rag=True, retriever_name="MedCPT", corpus_name="RareDisease", hf_access_token="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")

medrag.evaluate(snippetsNumber=7)