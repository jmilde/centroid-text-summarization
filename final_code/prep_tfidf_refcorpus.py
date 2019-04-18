import re
from tqdm import tqdm
from nltk import word_tokenize, sent_tokenize
import os
from nltk.corpus import stopwords
import string

def save_txt(filename, lines, split=""):
    """writes lines to text file."""
    with open(filename, 'w') as file:
        for line in lines:
            print(line+split, file= file)

def prep():
    ref_docs=[]
    path_refs = "../data/cnn_stories_tokenized"
    topic_path, _, files = next(os.walk(path_refs))

    for fl in tqdm(files):
        txt = re.sub('\s\s+', " ",
                     re.sub("\n", " ",
                            re.search(r"(.|\n)*?(?=@highlight)", open(topic_path+"/"+fl).read()).group()))
        ref_docs.append(txt.lower())

    save_txt("../data/ref_docs.txt", ref_docs, split="")

    ### cleaned version
    ref_docs = open("../data/ref_docs.txt").read().split("\n")

    remove = set(stopwords.words("english"))
    remove.update(list(string.punctuation))

    ref_docs_clean = []
    for doc in tqdm(ref_docs):
        ref_docs_clean.append(" ".join(
            [" ".join([word for word in word_tokenize(sent.lower()) if word not in remove])
             for sent in sent_tokenize(doc)]))
    save_txt("../data/ref_docs_clean.txt", ref_docs_clean, split="")

if __name__ == "__main__":
    prep()
