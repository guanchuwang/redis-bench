CUDA_VISIBLE_DEVICES=0 python zero-shot-bench/src/zero_shot_exp.py --model_id mistralai/Mistral-7B-Instruct-v0.2 
CUDA_VISIBLE_DEVICES=1 python zero-shot-bench/src/zero_shot_exp.py --model_id meta-llama/Llama-2-7b-chat-hf
CUDA_VISIBLE_DEVICES=2 python zero-shot-bench/src/zero_shot_exp.py --model_id google/gemma-1.1-7b-it
CUDA_VISIBLE_DEVICES=3 python zero-shot-bench/src/zero_shot_exp.py --model_id Qwen/Qwen2-7B-Instruct
CUDA_VISIBLE_DEVICES=0 python zero-shot-bench/src/zero_shot_exp.py --model_id microsoft/Phi-3-small-8k-instruct

# If you did not log in to the Hugging Face Hub, you can use add access_token to command line arguments
# Example: python zero-shot-bench/src/zero_shot_exp.py --model_id mistralai/Mistral-7B-Instruct-v0.2 --access_token <access_token>