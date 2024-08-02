import os
import tqdm
import json
import regex as re
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter

def ends_with_ending_punctuation(s):
    ending_punctuation = ('.', '?', '!')
    return any(s.endswith(char) for char in ending_punctuation)

def concat(title, content):
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + content.strip()
    else:
        return title.strip() + ". " + content.strip()

if __name__ == "__main__":
    dat = load_dataset("wikipedia", "20220301.en", cache_dir="./corpus/wikipedia", trust_remote_code=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    if not os.path.exists("corpus/wikipedia/chunk"):
        os.makedirs("corpus/wikipedia/chunk")
    
    batch_size = 10000
    len_just = len(str(len(dat['train']) // batch_size + 1))

    saved_text = []
    for i in tqdm.tqdm(range(len(dat['train']))):
        save_id = i // batch_size
        if os.path.exists("corpus/wikipedia/chunk/wiki20220301en{:s}.jsonl".format(str(save_id).rjust(len_just, '0'))):
            continue
        texts = text_splitter.split_text(dat['train'][i]['text'].strip())
        curr_text = [json.dumps({"id": '_'.join([dat['train'][i]['id'], str(j)]), "title": dat['train'][i]['title'], "content": re.sub("\s+", " ", t), "contents": concat(dat['train'][i]['title'], re.sub("\s+", " ", t))}) for j, t in enumerate(texts)]
        saved_text.extend(curr_text)
        if (i + 1) % batch_size == 0:
            with open("corpus/wikipedia/chunk/wiki20220301en{:s}.jsonl".format(str(save_id).rjust(len_just, '0')), 'w') as f:
                f.write('\n'.join(saved_text))
            saved_text = []
    if len(saved_text) > 0:
        with open("corpus/wikipedia/chunk/wiki20220301en{:s}.jsonl".format(str(save_id).rjust(len_just, '0')), 'w') as f:
            f.write('\n'.join(saved_text))