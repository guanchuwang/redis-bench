import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import sys
from datasets import load_from_disk
from sklearn.metrics import accuracy_score
from tqdm import tqdm




class LLM_Agent:
    def __init__(self, model_id = "mistralai/Mistral-7B-Instruct-v0.2", access_token = "Your HF Token here"):
        self.model_id = model_id
        tokenizer_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, token=access_token, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            token=access_token,
            trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        self.A_ids = self.tokenizer("A").input_ids[-1]
        self.B_ids = self.tokenizer("B").input_ids[-1]
        self.C_ids = self.tokenizer("C").input_ids[-1]
        self.D_ids = self.tokenizer("D").input_ids[-1]
        self.system_prompt = "You are a helpful medical expert, and your task is to answer a multi-choice medical question. Please first choose the answer from the provided options and then provide the explanation."
       
       

    @torch.no_grad()
    def output_logit(self, input_text, **kwargs):
        inputs = self.tokenizer(input_text, padding=False, return_tensors="pt")
        input_ids = inputs.input_ids.cuda()
        logits = self.llm(input_ids=input_ids).logits[:, -1].flatten()
        probs = torch.nn.functional.softmax(
            torch.tensor([
                logits[self.A_ids],
                logits[self.B_ids],
                logits[self.C_ids],
                logits[self.D_ids],
            ]), dim=0)
        return probs

    def num_tokens(self, text):
        encoding = self.tokenizer(text, padding=False, return_tensors="pt")
        return encoding.input_ids.shape[-1]


    def answermultiplechoice(self, question_doc):
        if "gemma-1" in self.model_id.lower():
            prompt = [{"role": "user", "content": question_doc["input"][:-7]}] + \
                 [{"role": "assistant", "content": "Answer: "}]
            prompt_in_template = self.tokenizer.apply_chat_template(prompt, tokenize=False)[:-len("<end_of_turn>\n")]
        
        elif "phi" in self.model_id.lower():
            prompt = [{"role": "user", "content": question_doc["input"][:-7]}] + \
                    [{"role": "assistant", "content": "Answer: "}]
            prompt_in_template = self.tokenizer.apply_chat_template(prompt, tokenize=False)[len("<|endoftext|>"):-len(f"<|end|>\n<|endoftext|>\n")]

        elif "qwen" in self.model_id.lower():
            prompt = [{"role": "user", "content": question_doc["input"][:-7]}] + \
                    [{"role": "assistant", "content": "Answer: "}]
            prompt_in_template = self.tokenizer.apply_chat_template(prompt, tokenize=False)[:-len(f"<|im_end|>\n")]

        elif "llama-2" in self.model_id.lower(): 
            prompt = [{"role": "user", "content": question_doc["input"][:-7]}] + \
                    [{"role": "assistant", "content": "Answer: "}]
            prompt_in_template = self.tokenizer.apply_chat_template(prompt, tokenize=False)[:-len(f"</s>")]
            
        elif "mistral" in self.model_id.lower():
            prompt = [{"role": "user", "content": question_doc["input"][:-7]}] + \
                    [{"role": "assistant", "content": "Answer: "}]
            prompt_in_template = self.tokenizer.apply_chat_template(prompt, tokenize=False)[:-len("</s>")]

        probs = self.output_logit(prompt_in_template)
        answer = int(probs.argmax(-1))
        return answer, probs.detach().numpy()

    def evaluate(self, dataset_dir):
        eval_dataset = load_from_disk(f"{dataset_dir}")        
        dataset_size = len(eval_dataset)
        llm_ans_buf = []

        for ques_idx in tqdm(range(dataset_size)):
            question_doc = eval_dataset[ques_idx]
            answer, probs = self.answermultiplechoice(question_doc)
            llm_ans_buf.append(answer)

        acc_score = accuracy_score(eval_dataset["cop"], llm_ans_buf)
        results = {
            "model": self.llm.config.name_or_path,
            "dataset": "ReDisQA",
            "use_rag": False,
            "accuracy": acc_score,
            "pred_ans": llm_ans_buf,
            "golden_ans": eval_dataset["cop"],
        }

        model_name = self.llm.config.name_or_path.replace("/", "-")
        results_fname = f"./model_{model_name}_dataset_ReDisQA_withoutRAG.json"
        
        # Construct the relative path to the results directory
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
        results_dir = os.path.join(script_dir, "..", "results")  # Going one level up to project root and then to results directory
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)  # Create the results directory if it does not exist

        results_path = os.path.join(results_dir, results_fname)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        return acc_score


