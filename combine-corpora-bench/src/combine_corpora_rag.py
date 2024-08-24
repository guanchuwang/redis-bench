import sys
import os
import scipy.stats as stats
import numpy as np
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import json

# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the paths to the other directories
meta_data_path = os.path.abspath(os.path.join(current_dir, '../../meta-data-bench/src'))
rag_bench_path = os.path.abspath(os.path.join(current_dir, '../../rag-bench/src'))

# Append these paths to sys.path
sys.path.append(meta_data_path)
sys.path.append(rag_bench_path)

# Now import the necessary modules
from MetaDataRAG import MetaDataRAG
from medrag import MedRAG

class CombineCorporaRAG:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2", combined_retriever_name="MedCPT", combined_corpus_name="Textbooks", hf_access_token="Your HF access token"):
        self.combined_medrag = MedRAG(model_id=model_id, retriever_name=combined_retriever_name, corpus_name=combined_corpus_name,  hf_access_token=hf_access_token)
        self.metadata_rag = MetaDataRAG(model_id=model_id, access_token=hf_access_token)
        self.model_id = model_id
        self.combined_retriever_name = combined_retriever_name
        self.combined_corpus_name = combined_corpus_name


    def answermultiplechoice(self, question_doc, k=5):
        medrag_answer, medrag_retrieved_snippets, medrag_scores, medrag_prob = self.combined_medrag.answermultiplechoice(question_doc, k)
        medrag_prob_entropy = stats.entropy(medrag_prob, axis=0)
        
        metadata_answer, metadata_prob = self.metadata_rag.answermultiplechoice(question_doc, k)
        metadata_prob_entropy = stats.entropy(metadata_prob, axis=0)

        if metadata_prob_entropy < medrag_prob_entropy:
            return metadata_answer
        else:
            return medrag_answer
        
    def evaluate(self, dataset_name = "ReDisQA", snippetsNumber = 7):
        eval_dataset = load_dataset("guan-wang/ReDis-QA")['test'] 
        dataset_size = len(eval_dataset["cop"])
        llm_ans_buf = []
        
      
        for ques_idx in tqdm(range(dataset_size)):
            question_doc = eval_dataset[ques_idx]
          
            answer = self.answermultiplechoice(question_doc=question_doc, k=snippetsNumber)
            llm_ans_buf.append(answer)
            
    
        # calculate the accuracy
        acc_score = accuracy_score(eval_dataset["cop"][:dataset_size], llm_ans_buf)

        
        results = {
            "model": self.model_id,
            "dataset": dataset_name,
            "corpus": self.combined_corpus_name + "+ReCOP",
            "retriever": self.combined_retriever_name,
            "use_rag": "yes",
            "snippetsNumber": snippetsNumber,
            "accuracy": acc_score,
            "pred_ans": llm_ans_buf,
            "golden_ans": eval_dataset["cop"],
        }

        model_name = self.model_id.replace("/", "-")
        results_fname = f"model_{model_name}_dataset_{dataset_name}_corpus_ReCOP+{self.combined_corpus_name}_retriever_{self.combined_retriever_name}_snippetNum_{snippetsNumber}.json"
        
        # Construct the relative path to the results directory
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
        results_dir = os.path.join(script_dir, "..", "results")  # Going one level up to project root and then to results directory
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)  # Create the results directory if it does not exist

        results_path = os.path.join(results_dir, results_fname)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        return acc_score

