CUDA_VISIBLE_DEVICES=2 python meta-data-bench/src/meta_data_rag_exp.py --k 7 --model_id mistralai/Mistral-7B-Instruct-v0.2 
CUDA_VISIBLE_DEVICES=0 python meta-data-bench/src/meta_data_rag_exp.py --k 7 --model_id meta-llama/Llama-2-7b-chat-hf
CUDA_VISIBLE_DEVICES=0 python meta-data-bench/src/meta_data_rag_exp.py --k 7 --model_id google/gemma-1.1-7b-it
CUDA_VISIBLE_DEVICES=0 python meta-data-bench/src/meta_data_rag_exp.py --k 7 --model_id Qwen/Qwen2-7B-Instruct 
CUDA_VISIBLE_DEVICES=0 python meta-data-bench/src/meta_data_rag_exp.py --k 7 --model_id microsoft/Phi-3-small-8k-instruct