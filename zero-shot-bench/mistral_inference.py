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

    def __init__(self, llm, tokenizer, batch_size=25):
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
        # ipdb.set_trace()
        inputs = self.tokenizer(input_text, padding=False, return_tensors="pt")
        input_ids = inputs.input_ids.cuda()
        logits = self.llm(
            input_ids=input_ids,
        ).logits[:, -1].flatten()
        # ipdb.set_trace()

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



# sys_prompt = 'You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''
# sys_prompt = 'You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first choose the answer from the provided options and then provide the explanation. Organize your output in a json formatted as Dict{"answer_choice": Str{A/B/C/...}, "explanation": Str(explanation)}.'''
sys_prompt = 'You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first choose the answer from the provided options and then provide the explanation.'

# parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('--batch_size', default=16, type=int)


def main():

    # args = parser.parse_args()
    access_token = "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"
    dataset_name = "ReDisQA_v2" # "medmcqa_rare_disease_v2"

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"

    tokenizer_id = model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, cache_dir="/scratch", token=access_token, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
        token=access_token,
        cache_dir="/scratch",
        trust_remote_code=True
    )

    assistance = LLM_Agent(model, tokenizer, batch_size=1)

    # corpus = load_from_disk(f"../corpus/data/nord_corpus/")
    # corpus_contents = np.array(corpus["contents"])
    # corpus_title = np.array(corpus["title"])
    # retrieved_doc_num = 7

    eval_dataset = load_from_disk(f"../data_clean/data/{dataset_name}")
    dataset_size = len(eval_dataset)
    llm_ans_buf = []

    for ques_idx in tqdm(range(dataset_size)):
        question_doc = eval_dataset[ques_idx ]

        prompt = [{"role": "user", "content": question_doc["input"][:-7]}] + \
                 [{"role": "assistant", "content": "Answer: "}]
        prompt_in_template = assistance.tokenizer.apply_chat_template(prompt, tokenize=False)[:-len("</s>")]
        # print(prompt_in_template)
        # ipdb.set_trace()

        # prompt_in_template = "<|user|>\n" + question_doc["input"][:-7] + "<|end|>\n<|assistant|>\n" + "Answer: "
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
        "corpus": None,
        "retriever": None,
        "accuracy": acc_score,
        "pred_ans": llm_ans_buf,
        "golden_ans": eval_dataset["cop"],
    }

    model_name = model_id.replace("/", "-")
    results_fname = f"./model_{model_name}_dataset_{dataset_name}_corpus_None_retriever_None.json"
    with open(f"./results/{results_fname}", "w") as f:
        json.dump(results, f, indent=4)



if __name__ == "__main__":
    main()




