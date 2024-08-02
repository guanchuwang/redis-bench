import os
import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .utils import RetrievalSystem
from datasets import load_from_disk
from sklearn.metrics import accuracy_score




class MedRAG:

    def __init__(self, llm_name=None, rag=True, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus", cache_dir=None, hf_access_token="Your HF access token"):
        self.llm_name = llm_name
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.access_token = hf_access_token
        self.system_prompt_without_rag = "You are a helpful medical expert, and your task is to answer a multi-choice medical question. Please first choose the answer from the provided options and then provide the explanation."
        self.system_prompt_with_rag = "You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first choose the answer from the provided options and then provide the explanation."

        if rag:
            self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir)
           
        else:
            self.retrieval_system = None


        
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir, token=self.access_token, trust_remote_code=True)
        self.tokenizer.padding_side = "left"

        if "mistral" in llm_name.lower():
            self.max_length = 32678
            self.context_length = 32678
        elif "llama-3" in llm_name.lower():
            self.max_length = 8192
            self.context_length = 8192
        elif "llama-2" in llm_name.lower():
            self.max_length = 4096
            self.context_length = 4096
        elif "gemma-2" in llm_name.lower():
            self.max_length = 8192
            self.context_length = 8192
        elif "gemma-1" in llm_name.lower():
            self.max_length = 8192
            self.context_length = 8192
        elif "qwen" in llm_name.lower():
            self.max_length = 131072
            self.context_length = 131072
        elif "phi-3" in llm_name.lower():
            self.max_length = 8192
            self.context_length = 8192

        self.llm = AutoModelForCausalLM.from_pretrained(
                    self.llm_name,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    token=self.access_token,
                    trust_remote_code=True
                )
        

        self.A_ids = self.tokenizer("A").input_ids[-1]  # 
        self.B_ids = self.tokenizer("B").input_ids[-1]  #
        self.C_ids = self.tokenizer("C").input_ids[-1]  #
        self.D_ids = self.tokenizer("D").input_ids[-1]  #


    @torch.no_grad()
    def output_string(self, input_text, **kwargs):

        inputs = self.tokenizer(input_text, padding=False, return_tensors="pt")
        input_ids = inputs.input_ids.cuda()
        ans_ids = self.llm.generate(input_ids, do_sample=False, max_new_tokens=32)
        ans_str = self.tokenizer.batch_decode(ans_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False)[0]

        return ans_str

    @torch.no_grad()
    def output_logit(self, input_text, **kwargs):
        inputs = self.tokenizer(input_text, padding=False, return_tensors="pt")
        input_ids = inputs.input_ids.cuda()

        logits = self.llm(
            input_ids=input_ids,
        ).logits[:, -1].flatten()  # take the last token logits (the predicted token)


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
        )

        return probs
    
    def num_tokens(self, text):
        """Return the number of tokens in a string."""
        # ipdb.set_trace()
        encoding = self.tokenizer(text, padding=False, return_tensors="pt")
        return encoding.input_ids.shape[-1]
    


    def answermultiplechoice(self, question_doc, k=5):
        '''
        question (str): question to be answered
        k (int): number of snippets to retrieve
        '''
        question = question_doc["input"]
        keyword = question_doc["rare disease"]
        context = ""
        contexts_valid = []
        # retrieve relevant snippets
        if self.rag:
            retrieved_snippets, scores = self.retrieval_system.retrieve(question, k=k)
            contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, retrieved_snippets[idx]["title"], retrieved_snippets[idx]["content"]) for idx in range(len(retrieved_snippets))]
    
            contexts_valid = []
            if len(contexts) == 0:
                contexts_valid = [""]              
            else:
                for idx, doc in enumerate(contexts):
                    token_num = self.num_tokens("\n".join(contexts[:idx+1]))
                    if token_num > (self.context_length):
                        break
                    contexts_valid.insert(0,doc)
            context = "\n\n".join([doc for doc in contexts_valid])
             
        else:
            retrieved_snippets = []
            scores = []
            

        # generate answers
        if not self.rag:
            if "gemma-1" in self.llm_name.lower():
                prompt = [{"role": "user", "content": question[:-7]}] + \
                [{"role": "assistant", "content": "Answer: <b>"}]
                prompt_in_template = self.tokenizer.apply_chat_template(prompt, tokenize=False)[:-len("<end_of_turn>\n")]

            elif "phi" in self.llm_name.lower():
                prompt = [{"role": "system", "content": self.system_prompt_without_rag}] + \
                [{"role": "user", "content": question[:-7]}] + \
                [{"role": "assistant", "content": "Answer: "}]
                prompt_in_template = self.tokenizer.apply_chat_template(prompt, tokenize=False)[len("<|endoftext|>"):-len(f"<|end|>\n<|endoftext|>\n")]
                
            elif "qwen" in self.llm_name.lower():
                prompt = [{"role": "system", "content": self.system_prompt_without_rag}] + \
                [{"role": "user", "content": question[:-7]}] + \
                [{"role": "assistant", "content": "Answer: "}]
                prompt_in_template = self.tokenizer.apply_chat_template(prompt, tokenize=False)[:-len(f"<|im_end|>\n")]
                

            elif "llama-2" in self.llm_name.lower():
                prompt = [{"role": "system", "content": self.system_prompt_without_rag}] + \
                [{"role": "user", "content": question[:-7]}] + \
                [{"role": "assistant", "content": "Answer: "}]
                prompt_in_template = self.tokenizer.apply_chat_template(prompt, tokenize=False)[:-len(f"</s>")]

            elif "mistral" in self.llm_name.lower():
                prompt = [{"role": "system", "content": self.system_prompt_without_rag}] + \
                [{"role": "user", "content": question[:-7]}] + \
                [{"role": "assistant", "content": "Answer: "}]
                prompt_in_template = self.tokenizer.apply_chat_template(prompt, tokenize=False)[:-len("</s>")]


        else:
            if "gemma-1" in self.llm_name.lower():
                prompt_in_template = "<bos><start_of_turn>system\n" + self.system_prompt_with_rag + "<end_of_turn>\n" + \
                        "<start_of_turn>user\n" + context + "<end_of_turn>\n" + \
                        "<start_of_turn>user\n" + question[:-7] + "<end_of_turn>\n" + "<start_of_turn>model\nAnswer: <b>"

            elif "phi" in self.llm_name.lower():
                prompt = [{"role": "system", "content": self.system_prompt_with_rag}] + \
                [{"role": "user", "content": context}] + \
                [{"role": "user", "content": question[:-7]}] + \
                [{"role": "assistant", "content": "Answer: "}]
                prompt_in_template = self.tokenizer.apply_chat_template(prompt, tokenize=False)[len("<|endoftext|>"):-len(f"<|end|>\n<|endoftext|>\n")]
            
            elif "qwen" in self.llm_name.lower():
                prompt = [{"role": "system", "content": self.system_prompt_with_rag}] + \
                [{"role": "user", "content": context}] + \
                [{"role": "user", "content": question[:-7]}] + \
                [{"role": "assistant", "content": "Answer: "}]
                prompt_in_template = self.tokenizer.apply_chat_template(prompt, tokenize=False)[:-len(f"<|im_end|>\n")]
                
            elif "llama-2" in self.llm_name.lower():
                prompt_in_template = "<s>[INST] <<SYS>>\n" + self.system_prompt_with_rag + "\n<</SYS>>" + \
                                    context + \
                                     "<|user|>\n" + question[:-7] + " [/INST] Answer: "
            
            elif "mistral" in self.llm_name.lower():
                prompt_in_template = "<s>[INST] " + self.system_prompt_with_rag + " [/INST] \n[INST]" + \
                             context + \
                             " [/INST] \n[INST]" + question[:-7] + " [/INST] \nAnswer: "


        prob = self.output_logit(prompt_in_template)
        answer = int(prob.argmax(-1))
        return answer, retrieved_snippets, scores

    
 

    def evaluate(self, dataset_name = "ReDisQA_v2", snippetsNumber = 7):
        eval_dataset = load_from_disk(f"/home/gw22/python_project/rare_disease_dataset/data_clean/data/{dataset_name}")
        dataset_size = len(eval_dataset["cop"])
        llm_ans_buf = []
        
      
        for ques_idx in tqdm(range(dataset_size)):
            question_doc = eval_dataset[ques_idx]
          
            answer, snippets, scores= self.answermultiplechoice(question_doc=question_doc, k=snippetsNumber)

            llm_ans_buf.append(answer)
            
    
        # calculate the accuracy
        acc_score = accuracy_score(eval_dataset["cop"][:dataset_size], llm_ans_buf)

        
        results = {
            "model": self.llm_name,
            "dataset": dataset_name,
            "corpus": self.corpus_name,
            "retriever": self.retriever_name,
            "use_rag": "yes" if self.rag else "no",
            "snippetsNumber": snippetsNumber if self.rag else 0,
            "accuracy": acc_score,
            "pred_ans": llm_ans_buf,
            "golden_ans": eval_dataset["cop"],
        }

        model_name = self.llm_name.replace("/", "-")
        if self.rag:
            results_fname = f"model_{model_name}_dataset_{dataset_name}_corpus_{self.corpus_name}_retriever_{self.retriever_name}_snippetNum_{snippetsNumber}.json"
        
        else:
            results_fname = f"model_{model_name}_dataset_{dataset_name}_withoutRAG.json"

        with open(f"results/{results_fname}", "w") as f:
            json.dump(results, f, indent=4)

        return acc_score