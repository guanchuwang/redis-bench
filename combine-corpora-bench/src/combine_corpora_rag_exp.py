from combine_corpora_rag import CombineCorporaRAG
import argparse


def main():
    parser = argparse.ArgumentParser(description='Combined Corpora RAG Experiment')
    parser.add_argument('--k', default=7, type=int)
    parser.add_argument('--model_id', default="mistralai/Mistral-7B-Instruct-v0.2", type=str)
    parser.add_argument('--retriever_name', default="MedCPT", type=str)
    parser.add_argument('--corpus_name', default="Textbooks", type=str)
    parser.add_argument('--access_token', type=str)

    args = parser.parse_args()
    combined_rag = CombineCorporaRAG(model_id=args.model_id, combined_retriever_name=args.retriever_name, combined_corpus_name=args.corpus_name, hf_access_token=args.access_token)
    combined_rag.evaluate(snippetsNumber=args.k)

if __name__ == "__main__":
    main()