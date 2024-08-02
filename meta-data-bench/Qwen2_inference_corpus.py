import time, json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

# from openai import OpenAI
import sys
import re
import ipdb
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import default_data_collator
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import tiktoken

sys.path.append("./")


class LLM_Agent:

    def __init__(self, llm, tokenizer):
        self.tokenizer = tokenizer
        self.llm = llm
        self.tokenizer.padding_side = "left"
        self.A_ids = self.tokenizer("A").input_ids[-1]  #
        self.B_ids = self.tokenizer("B").input_ids[-1]  #
        self.C_ids = self.tokenizer("C").input_ids[-1]  #
        self.D_ids = self.tokenizer("D").input_ids[-1]  #

    @torch.no_grad()
    def output_json(self, input_text, **kwargs):

        inputs = self.tokenizer(input_text, padding=False, return_tensors="pt")
        input_ids = inputs.input_ids.cuda()
        ans_ids = self.llm.generate(input_ids, do_sample=False, max_new_tokens=32)
        ans_str = self.tokenizer.batch_decode(ans_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False)[0]
        # print(ans_str)
        # ipdb.set_trace()

        return ans_str

    @torch.no_grad()
    def output_logit(self, input_text, **kwargs):
        inputs = self.tokenizer(input_text, padding=False, return_tensors="pt")
        # ipdb.set_trace()
        input_ids = inputs.input_ids.cuda()
        logits = self.llm(
            input_ids=input_ids,
        ).logits[:, -1].flatten()

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[self.A_ids],
                        logits[self.B_ids],
                        logits[self.C_ids],
                        logits[self.D_ids],
                    ]
                ),
                dim=0,
            )
            # .detach()
            # .cpu()
            # .to(torch.float32)
            # .numpy()
        )

        return probs


def num_tokens(text, tokenizer):
    """Return the number of tokens in a string."""
    # ipdb.set_trace()
    encoding = tokenizer(text, padding=False, return_tensors="pt")
    return encoding.input_ids.shape[-1]

def doc_search(keyword, corpus_contents, corpus_title, tokenizer, retrieved_doc_num=2, token_budget=8192):

    # mask = [keyword[0] in title for title in corpus_title]
    mask = keyword[0] == corpus_title
    relevent_doc = corpus_contents[mask].tolist()
    relevent_doc_valid = []

    for idx, doc in enumerate(relevent_doc[:retrieved_doc_num]):
        token_num = num_tokens(" ".join(relevent_doc[:idx+1]), tokenizer)
        if token_num > token_budget:
            break
        # relevent_doc_valid.append(doc)
        relevent_doc_valid.insert(0, doc)

    # ipdb.set_trace()

    return relevent_doc_valid

sys_prompt = 'You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first choose the answer from the provided options and then provide the explanation.'

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--k', default=5, type=int)


def main():

    args = parser.parse_args()
    access_token = "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"
    dataset_name = "ReDisQA_v2" # "medmcqa_rare_disease_v2"

    model_id = "Qwen/Qwen2-7B-Instruct"

    tokenizer_id = model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, cache_dir="/scratch", token=access_token)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
        token=access_token,
        cache_dir="/scratch"
    )

    assistance = LLM_Agent(model, tokenizer)

    corpus = load_from_disk(f"../corpus/data/nord_corpus_v2/")
    corpus_contents = np.array(corpus["contents"])
    corpus_title = np.array(corpus["rare-disease"])
    retrieved_doc_num = args.k

    eval_dataset = load_from_disk(f"../data_clean/data/{dataset_name}")
    dataset_size = len(eval_dataset)
    llm_ans_buf = []

    for ques_idx in tqdm(range(dataset_size)):
        # if ques_idx < 507:
        #     continue

        question_doc = eval_dataset[ques_idx]
        keyword = question_doc["rare disease"]
        relevent_doc = doc_search(keyword, corpus_contents, corpus_title, tokenizer, retrieved_doc_num)

        prompt = [{"role": "system", "content": sys_prompt}] + \
                 [{"role": "user", "content": doc} for doc in relevent_doc] + \
                 [{"role": "user", "content": question_doc["input"][:-7]}] + \
                 [{"role": "assistant", "content": "Answer: "}]
        prompt_in_template = assistance.tokenizer.apply_chat_template(prompt, tokenize=False)[:-len(f"<|im_end|>\n")]

        # ans = assistance.output_json(prompt_in_template)
        prob = assistance.output_logit(prompt_in_template)
        llm_ans_buf.append(int(prob.argmax(-1)))

        # print(keyword)
        # print(ans)
        # print(np.array(eval_dataset["cop"])[ques_idx], int(prob.argmax(-1)))
        # ipdb.set_trace()

    acc_score = accuracy_score(eval_dataset["cop"], llm_ans_buf)
    results = {
        "model": model_id,
        "dataset": dataset_name,
        "corpus": "nord_v2",
        "retriever": "golden",
        "snippetsNumber": retrieved_doc_num,
        "accuracy": acc_score,
        "pred_ans": llm_ans_buf,
        "golden_ans": eval_dataset["cop"],
    }

    model_name = model_id.replace("/", "-")
    results_fname = f"./model_{model_name}_dataset_{dataset_name}_corpus_nordv2_retriever_metadata_k_{str(retrieved_doc_num)}.json"
    with open(f"./results/{results_fname}", "w") as f:
        json.dump(results, f, indent=4)



if __name__ == "__main__":
    main()




