import argparse
from MetaDataRAG import MetaDataRAG  # assuming the class is defined in a file named meta_data_rag.py

def main():
    parser = argparse.ArgumentParser(description='MetaDataRAG Experiment Script')
    parser.add_argument('--k', default=5, type=int, help='Number of documents to retrieve')
    parser.add_argument('--model_id', type=str, default='google/gemma-1.1-7b-it', help='ID of the model to use')
    parser.add_argument('--access_token', type=str, help='Hugging Face access token')
    args = parser.parse_args()

    agent = MetaDataRAG(
        model_id=args.model_id,
        access_token=args.access_token,
    )

    accuracy = agent.evaluate(retrieved_doc_num=args.k)
    print(f"Evaluation completed with accuracy: {accuracy}")

if __name__ == "__main__":
    main()