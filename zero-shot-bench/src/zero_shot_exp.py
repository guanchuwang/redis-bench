import argparse
from LLM_Agent import LLM_Agent  

def main():
    parser = argparse.ArgumentParser(description='Zero Shot Experiment Script')
    parser.add_argument('--dataset_dir', type=str, default="/home/gw22/python_project/rare_disease_dataset/data_clean/data/ReDisQA_v2", help='Directory of the dataset to evaluate')
    parser.add_argument('--model_id', type=str, default='google/gemma-1.1-7b-it', help='ID of the model to use')
    parser.add_argument('--access_token', type=str, help='Hugging Face access token')
    args = parser.parse_args()

    agent = LLM_Agent(
        model_id=args.model_id,
        access_token=args.access_token
    )

    accuracy = agent.evaluate(dataset_dir=args.dataset_dir)
    print(f"Evaluation completed with accuracy: {accuracy}")

if __name__ == "__main__":
    main()