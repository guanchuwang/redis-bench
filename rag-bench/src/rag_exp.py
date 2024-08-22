from medrag import MedRAG
import argparse


def main():
    parser = argparse.ArgumentParser(description='RAG Experiment')
    parser.add_argument('--k', default=5, type=int)
    parser.add_argument('--model_id', default="mistralai/Mistral-7B-Instruct-v0.2", type=str)
    parser.add_argument('--retriever_name', default="MedCPT", type=str)
    parser.add_argument('--corpus_name', default="Textbooks", type=str)
    parser.add_argument('--hf_access_token', type=str)

    args = parser.parse_args()
    medrag = MedRAG(model_id=args.model_id, retriever_name=args.retriever_name, corpus_name=args.corpus_name, hf_access_token=args.hf_access_token)
    medrag.evaluate(snippetsNumber=args.k)

if __name__ == "__main__":
    main()