# Centroid Text Summarizer
This is an improved centroid based Text Summarization.

## Getting Started

To get the summarizer to work one first has to install requiered libraries and run some preprocessings

### Prerequisites

Besides Python 3.x, following libraries are additionally requiered, run the following commands in your shell

```
pip install gensim
pip install nltk
pip install sklearn
pip install scipy
pip install numpy
pip install tqdm
```
The default for all following instructions is to have the project folder as cd.

First run the following commands
```
mkdir ./data
mkdir ./data/embed_files
```
In the next step download the following embeddings and if requiered, unpack them.
For news summarizations get the google news embeddings:
```
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
```

For other domains i recommend using glove embeddings which can be downloaded here:
```
nlp.stanford.edu/data/glove.6B.zip
```
If you pick glove embeddings make sure to run python3 in your shell and then run the following lines, to change the glove embeddings to the right format.
```
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec("./data/embed_files/glove.6B.300d.txt", "./data/embed_files/glove")
```

In order to have a reference corpus (for news) download the following file in ./data and unpack it there 
```
https://drive.google.com/uc?id=0BzQ6rtO2VN95cmNuc2xwUS1wdEE&export=download
```
Then preprocess this corpus by running the following scripts in the shell
```
 
 python3 /code/prep_tfidf_refcorpus.py
```

## Running the tests

To run the summarizer open the parameters.py file and give the path to the folder where the single or multidocument to be summarized are.
Each document should be a seperate plaintext file.
Then run
```
python3 code/summarize.py
```
