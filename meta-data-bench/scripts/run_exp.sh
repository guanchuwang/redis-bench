CUDA_VISIBLE_DEVICES=0 python phi_inference_corpus.py           --k 7
CUDA_VISIBLE_DEVICES=0 python Qwen2_inference_corpus.py         --k 7
CUDA_VISIBLE_DEVICES=0 python llama_inference_corpus.py         --k 7
CUDA_VISIBLE_DEVICES=0 python gemma_inference_corpus.py         --k 7
CUDA_VISIBLE_DEVICES=0 python mistral_inference_corpus.py       --k 7

