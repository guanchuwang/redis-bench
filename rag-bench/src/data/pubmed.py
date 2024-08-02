import os
import gzip
import tqdm
import json

def ends_with_ending_punctuation(s):
    ending_punctuation = ('.', '?', '!')
    return any(s.endswith(char) for char in ending_punctuation)

def concat(title, content):
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + content.strip()
    else:
        return title.strip() + ". " + content.strip()

def extract(gz_fpath):
    titles = []
    abstracts = []
    title = ""
    abs = ""
    ids = []

    for line in gzip.open(gz_fpath, 'rt').read().split('\n'):
        if line.strip() == "<Article>" or line.strip().startswith("<Article "):
            title = ""
            abs = ""
        elif line.strip() == "</Article>":
            if abs.strip() == "":
                continue
            titles.append(title)
            abstracts.append(abs)
            ids.append(id)
        if line.strip().startswith("<PMID"):
            id = line.strip().strip("</PMID>").split(">")[-1]        
        if line.strip().startswith("<ArticleTitle>"):
            title = line.strip()[14:-15]
        if line.strip().startswith("<AbstractText"):
            if len(abs) == 0: 
                abs += "".join(line.strip()[13:-15].split('>')[1:])
            else:
                abs += " "
                abs += "".join(line.strip()[13:-15].split('>')[1:])

    return titles, abstracts, ids

if __name__ == "__main__":
    fnames = sorted([fname for fname in os.listdir("corpus/pubmed/baseline") if fname.endswith("xml.gz")])
    
    if not os.path.exists("corpus/pubmed/chunk"):
        os.makedirs("corpus/pubmed/chunk")

    for fname in tqdm.tqdm(fnames):
        if os.path.exists("corpus/pubmed/chunk/{:s}".format(fname.replace(".xml.gz", ".jsonl"))):
            continue
        gz_fpath = os.path.join("corpus/pubmed/baseline", fname)
        titles, abstracts, ids = extract(gz_fpath)
        saved_text = [json.dumps({"id": "PMID:"+str(ids[i]), "title": titles[i], "content": abstracts[i], "contents": concat(titles[i], abstracts[i])}) for i in range(len(titles))]
        with open("corpus/pubmed/chunk/{:s}".format(fname.replace(".xml.gz", ".jsonl")), 'w') as f:
            f.write('\n'.join(saved_text))