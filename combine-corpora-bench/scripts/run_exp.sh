CUDA_VISIBLE_DEVICES=3 python combine-corpora-bench/src/combine_corpora_rag_exp.py --k 7 --model_id mistralai/Mistral-7B-Instruct-v0.2 --retriever_name MedCPT --corpus_name Textbooks 
CUDA_VISIBLE_DEVICES=0 python combine-corpora-bench/src/combine_corpora_rag_exp.py --k 7 --model_id mistralai/Mistral-7B-Instruct-v0.2 --retriever_name MedCPT --corpus_name StatPearls
CUDA_VISIBLE_DEVICES=3 python combine-corpora-bench/src/combine_corpora_rag_exp.py --k 7 --model_id microsoft/Phi-3-small-8k-instruct --retriever_name MedCPT --corpus_name Textbooks
