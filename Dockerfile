FROM python:3
LABEL maintainer="peter.bourgonje@dfki.de"


RUN apt-get -y update && \
    apt-get upgrade -y && \
    apt-get install -y python3-dev &&\
    apt-get update -y &&\
    apt-get install git -y



## install prerequisites
RUN pip3 install gensim nltk sklearn scipy numpy tqdm flask Flask-Cors

RUN mkdir /summ
#RUN mkdir /summ/python
WORKDIR /summ
RUN git clone https://github.com/jmilde/centroid-text-summarization
RUN cd centroid-text-summarization &&\
    mkdir data &&\
    mkdir data/embed_files

# embeddings, cnn stories and ref_docs_clean need to be in data folder on the host machine
COPY data/embed_files/GoogleNews-vectors-negative300.bin centroid-text-summarization/data/embed_files
#COPY data/cnn_stories_tokenized centroid-text-summarization/data/cnn_stories_tokenized
#COPY data/ref_docs_clean.txt centroid-text-summarization/data/ref_docs_clean.txt

RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords

#RUN export LC_ALL=C.UTF-8
#RUN export LANG=C.UTF-8

EXPOSE 5000

# to generate tf idf vals on the fly, uncomment the following
#WORKDIR /summ/centroid-text-summarization/
#RUN python3 final_code/prep_tfidf_refcorpus.py

# .sav files need to be on the host machine already (tfidf script takes a long time)
COPY data/count_vect.sav centroid-text-summarization/data
COPY data/df_vect.sav centroid-text-summarization/data

WORKDIR /summ/centroid-text-summarization/final_code
ENTRYPOINT FLASK_APP=flaskController.py flask run --host=0.0.0.0
#CMD ["/bin/bash"]
