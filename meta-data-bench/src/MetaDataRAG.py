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


# sys.path.append("./")

class MetaDataRAG:

    def __init__(self, model_id = "mistralai/Mistral-7B-Instruct-v0.2", access_token = "Your HF Token here", corpus_dir="../corpus/data/nord_corpus_v2"):
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
        self.system_prompt = 'You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first choose the answer from the provided options and then provide the explanation.'
       
        self.corpus = load_from_disk(f"{corpus_dir}")
        self.corpus_contents = np.array(self.corpus["contents"])
        self.corpus_title = np.array(self.corpus["rare-disease"])


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

    def doc_search(self, keyword, retrieved_doc_num=5, token_budget=8192):
        mask = keyword[0] == self.corpus_title
        relevant_doc = self.corpus_contents[mask].tolist()
        relevant_doc_valid = []

        for idx, doc in enumerate(relevant_doc[:retrieved_doc_num]):
            token_num = self.num_tokens(" ".join(relevant_doc[:idx+1]))
            if token_num > token_budget:
                break
            relevant_doc_valid.insert(0, doc)
        return relevant_doc_valid

    def answermultiplechoice(self, question_doc, retrieved_doc_num):
        keyword = question_doc["rare disease"]
        relevant_doc = self.doc_search(keyword, retrieved_doc_num)

        if "gemma-1" in self.model_id.lower():
            prompt_in_template = "<bos><start_of_turn>system\n" + self.system_prompt + "<end_of_turn>\n" + \
                             "".join(["<start_of_turn>user\n" + doc + "<end_of_turn>\n" for doc in relevant_doc]) + \
                             "<start_of_turn>user\n" + question_doc["input"][:-7] + "<end_of_turn>\n" + "<start_of_turn>model\nAnswer: "
        
        elif "phi" in self.model_id.lower():
            prompt = [{"role": "system", "content": self.system_prompt}] + \
                 [{"role": "user", "content": doc} for doc in relevant_doc] + \
                 [{"role": "user", "content": question_doc["input"][:-7]}] + \
                 [{"role": "assistant", "content": "Answer: "}]
            prompt_in_template = self.tokenizer.apply_chat_template(prompt, tokenize=False)[len("<|endoftext|>"):-len(f"<|end|>\n<|endoftext|>\n")]

        elif "qwen" in self.model_id.lower():
            prompt = [{"role": "system", "content": self.system_prompt}] + \
                [{"role": "user", "content": doc} for doc in relevant_doc] + \
                [{"role": "user", "content": question_doc["input"][:-7]}] + \
                [{"role": "assistant", "content": "Answer: "}]
            prompt_in_template = self.tokenizer.apply_chat_template(prompt, tokenize=False)[:-len(f"<|im_end|>\n")]

        elif "llama-2" in self.model_id.lower(): 
            prompt_in_template = "<s>[INST] <<SYS>>\n" + self.system_prompt + "\n<</SYS>>" + \
                        "\n\n".join([doc for doc in relevant_doc]) + \
                        "<|user|>\n" + question_doc["input"][:-7] + " [/INST] Answer: "
            
        elif "mistral" in self.model_id.lower():
            prompt_in_template = "<s>[INST] " + self.system_prompt + " [/INST] \n[INST]" + \
                        "\n\n".join([doc for doc in relevant_doc]) + \
                        " [/INST] \n[INST]" + question_doc["input"][:-7] + " [/INST] \nAnswer: "


        probs = self.output_logit(prompt_in_template)
        answer = int(probs.argmax(-1))
        return answer, probs.detach().numpy()

    def evaluate(self, dataset_dir, retrieved_doc_num):
        eval_dataset = load_from_disk(f"{dataset_dir}")        
        dataset_size = len(eval_dataset)
        llm_ans_buf = []

        for ques_idx in tqdm(range(dataset_size)):
            question_doc = eval_dataset[ques_idx]
            answer, probs = self.answermultiplechoice(question_doc, retrieved_doc_num)
            llm_ans_buf.append(answer)

        acc_score = accuracy_score(eval_dataset["cop"], llm_ans_buf)
        results = {
            "model": self.llm.config.name_or_path,
            "dataset": "ReDisQA",
            "corpus": "ReCOP",
            "use_rag": True,
            "retriever": "golden",
            "snippetsNumber": retrieved_doc_num,
            "accuracy": acc_score,
            "pred_ans": llm_ans_buf,
            "golden_ans": eval_dataset["cop"],
        }

        model_name = self.llm.config.name_or_path.replace("/", "-")
        results_fname = f"./model_{model_name}_dataset_ReDisQA_corpus_nord_retriever_metadata_k_{str(retrieved_doc_num)}.json"
        
        # Construct the relative path to the results directory
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
        results_dir = os.path.join(script_dir, "..", "results")  # Going one level up to project root and then to results directory
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)  # Create the results directory if it does not exist

        results_path = os.path.join(results_dir, results_fname)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        return acc_score


def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--k', default=5, type=int)
    args = parser.parse_args()

    access_token = "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"
    # dataset_name = "ReDisQA_v2"
    model_id = "google/gemma-1.1-7b-it"
    retrieved_doc_num = args.k
    # corpus_dir = "../corpus/data/nord_corpus_v2/"

    agent = MetaDataRAG(model_id, access_token)
    accuracy = agent.evaluate(retrieved_doc_num)
    print(f"Evaluation completed with accuracy: {accuracy}")


if __name__ == "__main__":
    main()
