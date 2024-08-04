import argparse
from MetaDataRAG import MetaDataRAG  # assuming the class is defined in a file named meta_data_rag.py

def main():
    parser = argparse.ArgumentParser(description='MetaDataRAG Experiment Script')
    parser.add_argument('--k', default=5, type=int, help='Number of documents to retrieve')
    parser.add_argument('--dataset_dir', type=str, default="/home/gw22/python_project/rare_disease_dataset/data_clean/data/ReDisQA_v2", help='Directory of the dataset to evaluate')
    parser.add_argument('--model_id', type=str, default='google/gemma-1.1-7b-it', help='ID of the model to use')
    parser.add_argument('--corpus_dir', type=str, default="/home/gw22/python_project/rare_disease_dataset/corpus/data/nord_corpus_v2", help='Directory of the corpus data')
    parser.add_argument('--access_token', type=str, default="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", help='Hugging Face access token')
    args = parser.parse_args()

    agent = MetaDataRAG(
        model_id=args.model_id,
        access_token=args.access_token,
        corpus_dir=args.corpus_dir
    )

    accuracy = agent.evaluate(dataset_dir=args.dataset_dir, retrieved_doc_num=args.k)
    print(f"Evaluation completed with accuracy: {accuracy}")

if __name__ == "__main__":
    main()