"""
This code is built upon the MedRAG codebase.

Original source:
@article{xiong2024benchmarking,
  title={Benchmarking retrieval-augmented generation for medicine},
  author={Xiong, Guangzhi and Jin, Qiao and Lu, Zhiyong and Zhang, Aidong},
  journal={arXiv preprint arXiv:2402.13178},
  year={2024}
}

GitHub repository:
https://github.com/Teddy-XiongGZ/MedRAG
"""

from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer
import os
import faiss
import json
import torch
import tqdm
import numpy as np
import time

corpus_names = {
    "PubMed": ["pubmed"],
    "Textbooks": ["textbooks"],
    "StatPearls": ["statpearls"],
    "Wikipedia": ["wikipedia"],
    "RareDisease": ["raredisease"]
}

retriever_names = {
    "BM25": ["bm25"],
    "Contriever": ["facebook/contriever"],
    "SPECTER": ["allenai/specter"],
    "MedCPT": ["ncbi/MedCPT-Query-Encoder"],
}

class CustomizeSentenceTransformer(SentenceTransformer): # change the default pooling "MEAN" to "CLS"

    def _load_auto_model(self, model_name_or_path):
        """
        Creates a simple Transformer + CLS Pooling model and returns the modules
        """
        print("No sentence-transformers model found with name {}. Creating a new one with CLS pooling.".format(model_name_or_path))
        transformer_model = Transformer(model_name_or_path)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), 'cls')
        return [transformer_model, pooling_model]


def embed(chunk_dir, index_dir, model_name, device="cuda", **kwarg):

    save_dir = os.path.join(index_dir, "embedding")
    
    if "contriever" in model_name:
        model = SentenceTransformer(model_name, device=device if torch.cuda.is_available() else "cpu")
    else:
        model = CustomizeSentenceTransformer(model_name, device=device if torch.cuda.is_available() else "cpu")

    model.eval()

    fnames = sorted([fname for fname in os.listdir(chunk_dir) if fname.endswith(".jsonl")])


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for fname in tqdm.tqdm(fnames):
            fpath = os.path.join(chunk_dir, fname)
            save_path = os.path.join(save_dir, fname.replace(".jsonl", ".npy"))
            if os.path.exists(save_path):
                continue
            if open(fpath).read().strip() == "":
                continue
            texts = [json.loads(item) for item in open(fpath).read().strip().split('\n')]
            if "specter" in model_name.lower():
                texts = [model.tokenizer.sep_token.join([item["title"], item["content"]]) for item in texts]
            elif "contriever" in model_name.lower():
                texts = [". ".join([item["title"], item["content"]]).replace('..', '.').replace("?.", "?") for item in texts]
            elif "medcpt" in model_name.lower():
                texts = [[item["title"], item["content"]] for item in texts]
            
            start = time.time()
            embed_chunks = model.encode(texts)
            np.save(save_path, embed_chunks)
        embed_chunks = model.encode([""], **kwarg)
    return embed_chunks.shape[-1]


def construct_index(index_dir, model_name, h_dim=768):

    with open(os.path.join(index_dir, "metadatas.jsonl"), 'w') as f:
        f.write("")
    
    if "specter" in model_name.lower():
        index = faiss.IndexFlatL2(h_dim)
    else:
        index = faiss.IndexFlatIP(h_dim)

    for fname in tqdm.tqdm(sorted(os.listdir(os.path.join(index_dir, "embedding")))):
        curr_embed = np.load(os.path.join(index_dir, "embedding", fname))
        index.add(curr_embed)
        with open(os.path.join(index_dir, "metadatas.jsonl"), 'a+') as f:
            f.write("\n".join([json.dumps({'index': i, 'source': fname.replace(".npy", "")}) for i in range(len(curr_embed))]) + '\n')

    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    return index




class Retriever: 

    def __init__(self, retriever_name="ncbi/MedCPT-Query-Encoder", corpus_name="textbooks", db_dir="./corpus", **kwarg):
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name

        self.db_dir = db_dir
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
        self.chunk_dir = os.path.join(self.db_dir, self.corpus_name, "chunk")

        # Clone the corpus from Huggingface if the chunk directory does not exist
        if not os.path.exists(self.chunk_dir):
            print("Cloning the {:s} corpus from Huggingface...".format(self.corpus_name))
            os.system("git clone https://huggingface.co/datasets/MedRAG/{:s} {:s}".format(corpus_name, os.path.join(self.db_dir, self.corpus_name)))
            if self.corpus_name == "statpearls":
                print("Downloading the statpearls corpus from NCBI bookshelf...")
                os.system("wget https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz -P {:s}".format(os.path.join(self.db_dir, self.corpus_name)))
                os.system("tar -xzvf {:s} -C {:s}".format(os.path.join(db_dir, self.corpus_name, "statpearls_NBK430685.tar.gz"), os.path.join(self.db_dir, self.corpus_name)))
                print("Chunking the statpearls corpus...")
                os.system("python src/data/statpearls.py")

        # Create the index directory
        self.index_dir = os.path.join(self.db_dir, self.corpus_name, "index", self.retriever_name.replace("Query-Encoder", "Article-Encoder"))

        if "bm25" in self.retriever_name.lower():
            from pyserini.search.lucene import LuceneSearcher
            self.metadatas = None
            self.embedding_function = None
            if os.path.exists(self.index_dir):
                self.index = LuceneSearcher(os.path.join(self.index_dir))
            else:
                os.system("python -m pyserini.index.lucene --collection JsonCollection --input {:s} --index {:s} --generator DefaultLuceneDocumentGenerator --threads 16".format(self.chunk_dir, self.index_dir))
                self.index = LuceneSearcher(os.path.join(self.index_dir))

        else:
            if os.path.exists(os.path.join(self.index_dir, "faiss.index")):
                self.index = faiss.read_index(os.path.join(self.index_dir, "faiss.index"))
                self.metadatas = [json.loads(line) for line in open(os.path.join(self.index_dir, "metadatas.jsonl")).read().strip().split('\n')]
            else:
                print("[In progress] Embedding the {:s} corpus with the {:s} retriever...".format(self.corpus_name, self.retriever_name.replace("Query-Encoder", "Article-Encoder")))
                h_dim = embed(chunk_dir=self.chunk_dir, index_dir=self.index_dir, model_name=self.retriever_name.replace("Query-Encoder", "Article-Encoder"), **kwarg)
                print("[In progress] Embedding finished! The dimension of the embeddings is {:d}.".format(h_dim))
                self.index = construct_index(index_dir=self.index_dir, model_name=self.retriever_name.replace("Query-Encoder", "Article-Encoder"), h_dim=h_dim)
                print("[Finished] Corpus indexing finished!")
                self.metadatas = [json.loads(line) for line in open(os.path.join(self.index_dir, "metadatas.jsonl")).read().strip().split('\n')]            
            if "contriever" in retriever_name.lower():
                self.embedding_function = SentenceTransformer(self.retriever_name, device="cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.embedding_function = CustomizeSentenceTransformer(self.retriever_name, device="cuda" if torch.cuda.is_available() else "cpu")
            self.embedding_function.eval()

    def get_relevant_documents(self, question, k=32, **kwarg):
        assert type(question) == str
        question = [question]

        if "bm25" in self.retriever_name.lower():
            res_ = [[]]
            hits = self.index.search(question[0], k=k)
            res_[0].append(np.array([h.score for h in hits]))
            #indices = [{"source": '_'.join(h.docid.split('_')[:-1]), "index": eval(h.docid.split('_')[-1])} for h in hits]

            # Adjust file naming logic here
            indices = []
            for h in hits:
                docid_parts = h.docid.split('_')
                if 'nord' in docid_parts[0]:
                    source_name = '_'.join(['nordid',docid_parts[1]])
                else:
                    source_name = '_'.join(docid_parts[:-1])
                indices.append({
                    "source": source_name,
                    "index": eval(docid_parts[-1])
                })
        else:
            with torch.no_grad():
                query_embed = self.embedding_function.encode(question, **kwarg)
            res_ = self.index.search(query_embed, k=k)
            indices = [self.metadatas[i] for i in res_[1][0]]

        texts = self.idx2txt(indices)
        scores = res_[0][0].tolist()
        
        return texts, scores

    def idx2txt(self, indices): # return List of Dict of str
        '''
        Input: List of Dict( {"source": str, "index": int} )
        Output: List of str
        '''
        return [json.loads(open(os.path.join(self.chunk_dir, i["source"]+".jsonl")).read().strip().split('\n')[i["index"]]) for i in indices]

class RetrievalSystem:

    def __init__(self, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus"):
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        assert self.corpus_name in corpus_names
        assert self.retriever_name in retriever_names
        self.retrievers = []
        for retriever in retriever_names[self.retriever_name]:
            self.retrievers.append([])
            for corpus in corpus_names[self.corpus_name]:
                self.retrievers[-1].append(Retriever(retriever, corpus, db_dir))
    
    def retrieve(self, question, k=32, rrf_k=100):
        '''
            Given questions, return the relevant snippets from the corpus
        '''
        assert type(question) == str

        texts = []
        scores = []

        if "RRF" in self.retriever_name:
            k_ = max(k * 2, 100)
        else:
            k_ = k
        for i in range(len(retriever_names[self.retriever_name])):
            texts.append([])
            scores.append([])
            for j in range(len(corpus_names[self.corpus_name])):
                t, s = self.retrievers[i][j].get_relevant_documents(question, k=k_)
                texts[-1].append(t)
                scores[-1].append(s)
        texts, scores = self.merge(texts, scores, k=k, rrf_k=rrf_k)

        return texts, scores

    def merge(self, texts, scores, k=32, rrf_k=100):
        '''
            Merge the texts and scores from different retrievers
        '''
        RRF_dict = {}
        for i in range(len(retriever_names[self.retriever_name])):
            texts_all, scores_all = None, None
            for j in range(len(corpus_names[self.corpus_name])):
                if texts_all is None:
                    texts_all = texts[i][j]
                    scores_all = scores[i][j]
                else:
                    texts_all = texts_all + texts[i][j]
                    scores_all = scores_all + scores[i][j]
            if "specter" in retriever_names[self.retriever_name][i].lower():
                sorted_index = np.array(scores_all).argsort()
            else:
                sorted_index = np.array(scores_all).argsort()[::-1]
            texts[i] = [texts_all[i] for i in sorted_index]
            scores[i] = [scores_all[i] for i in sorted_index]
            for j, item in enumerate(texts[i]):
                if item["id"] in RRF_dict:
                    RRF_dict[item["id"]]["score"] += 1 / (rrf_k + j + 1)
                    RRF_dict[item["id"]]["count"] += 1
                else:
                    RRF_dict[item["id"]] = {
                        "id": item["id"],
                        "title": item["title"],
                        "content": item["content"],
                        "score": 1 / (rrf_k + j + 1),
                        "count": 1
                        }
        RRF_list = sorted(RRF_dict.items(), key=lambda x: x[1]["score"], reverse=True)
        if len(texts) == 1:
            texts = texts[0][:k]
            scores = scores[0][:k]
        else:
            texts = [dict((key, item[1][key]) for key in ("id", "title", "content")) for item in RRF_list[:k]]
            scores = [item[1]["score"] for item in RRF_list[:k]]
        return texts, scores