from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.spatial.distance import cosine
import pickle
import numpy as np
from nltk import word_tokenize, sent_tokenize
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import re
from tqdm import tqdm

def similarity(v1, v2):
    score = 0.0
    if np.count_nonzero(v1) != 0 and np.count_nonzero(v2) != 0:
        score = ((1 - cosine(v1, v2)) + 1) / 2
    return score


def load_embedding(language, topic):
    assert language=="english" or language=="german", "language needs to be 'german' or 'english'"
    assert topic=="general" or topic=="news", "topic needs to be 'general' or 'news'"
    if language == "english":
        if topic == "news":
            # google news
            model = KeyedVectors.load_word2vec_format('../data/embed_files/GoogleNews-vectors-negative300.bin', binary=True, limit=100000)
        elif topic == "general":
            # glove
            #glove2word2vec("../data/embed_files/glove.6B.300d.txt", "../data/embed_files/glove") #preprocess glove file to fit gensims word2vec format
            model = KeyedVectors.load_word2vec_format('../data/embed_files/glove', limit=100000)
    elif language=="german":
        if topic == "general":
            print("download pretrained general ones from here https://deepset.ai/german-word-embeddings")
        elif topic == "news":
            print("need to be trained on data of our partners")
    return model


def clean_txt(txt, remove):
    sents = sent_tokenize(txt)y
    clean_sents = [" ".join([word for word in word_tokenize(sent.lower()) if word not in remove])
                   for sent in sents]
    clean_txt = " ".join(clean_sents)
    raw_sents = [sent for sent in sents]
    return clean_txt, raw_sents


def topic_words(sents, model_path, topic_threshold, load=True, save=False, refdoc_path=None):
    if load:
        count_vect = pickle.load(open(model_path + "/count_vect.sav", 'rb'))
        doc_freq = pickle.load(open(model_path + "/df_vect.sav", 'rb'))
    else:
        assert refdoc_path is not None, "need to give the path of the cleaned reference corpus"
        ### get topic words via TF-IDF
        count_vect = CountVectorizer() #todo check settings
        ### IDF based on big reference corpus
        ref_docs = open(refdoc_path).read().split("\n")
        doc_freq = count_vect.fit_transform(ref_docs+sents)
        if save: ## to save trained models
            pickle.dump(count_vect, open(model_path + "/count_vect.sav", 'wb'))
            pickle.dump(doc_freq, open(model_path + "/df_vect.sav", 'wb'))

    feature_names = count_vect.get_feature_names()
    # add the doc freq to the tfidf class
    tfidf = TfidfTransformer(use_idf=True, smooth_idf=True).fit(doc_freq)

    # caluclate the tfidf scores for the input text
    tfidf_vector = tfidf.transform(count_vect.transform([sents]))
    coo_matrix = tfidf_vector.tocoo()
    tuples = zip(coo_matrix.col, coo_matrix.data)
    sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    #centroid_words = [feature_names[idx] for idx, score in sorted_items if score>topic_threshold]
    centroid_words_weights = {feature_names[idx]:score
                          for idx, score in sorted_items if score>topic_threshold}
    return centroid_words_weights, sorted_items, feature_names


def weight_sentences(txt, centroid_words_weights, tfidf_scores, feature_names, remove):
    d_score = []
    for doc in txt:
        # general tfidf weights
        tfidf_weights = {feature_names[idx]:score for idx, score in tfidf_scores}
        s_score = []
        centroid_weights = centroid_words_weights.copy()
        for i, sent in enumerate(sent_tokenize(doc)):
            score = []
            for wrd in word_tokenize(sent.lower()):
                if wrd not in remove and tfidf_weights.__contains__(wrd):
                    # if wrd is a centroid word that appears for the first time, give it a higher weight
                    if centroid_weights.__contains__(wrd):
                        score.append(centroid_weights[wrd]*3)
                        del centroid_weights[wrd]
                    else:
                        score.append(tfidf_weights[wrd])
            if len(score)>0:
                s_score.append([sum(score)/len(score),i]) # average the sentence score
        d_score.append(sorted(s_score, reverse=True))
    return d_score


# GET CENTROID VECTOR
def get_centroid(centroid_words, model):
    dim = model.vector_size
    centroid_vector = np.zeros(dim)
    count=0
    for idx, word in enumerate(centroid_words):
        if model.__contains__(word):
            centroid_vector = centroid_vector + model[word]
            count += 1
    if count>0:
        centroid_vector = np.divide(centroid_vector, count)
    return centroid_vector


def select_ntop(txt, scores, n_top, remove):
    clean_sents, raw_sents = [], []
    for doc, scr in zip(txt, scores):
        sel_sents = set([s[1] for i,s in enumerate(scr) if i<=n_top-1])
        for i,sent in enumerate(sent_tokenize(doc)):
            if i in sel_sents:
                raw_sents.append(sent)
                clean_sents.append(" ".join([wrd for wrd in word_tokenize(sent.lower()) if wrd not in remove]))
    return clean_sents, raw_sents


def score_sentences(sents, raw_sents, model, centroid_vector):
    dim = model.vector_size
    sentence_scores = []
    for i, sent in enumerate(sents):
        sent_vector = np.zeros(dim)
        count=0
        words = sent.split()
        for w in words:
            if model.__contains__(w):
                sent_vector = sent_vector + model[w]
                count += 1
        if count>0:
            sent_vector = np.divide(sent_vector, count)
        score = similarity(sent_vector, centroid_vector)
        sentence_scores.append((i, raw_sents[i], score, sent_vector))

    # rank sentences by score
    sentence_scores_sort = sorted(sentence_scores, key=lambda el: el[2], reverse=True)
    return sentence_scores_sort


def select_sentences(sentence_scores, sim_threshold, limit_type, limit, reorder):
    assert limit_type == "words" or limit_type == "bytes", "limit_type has to be 'words' or 'bytes'"
    count = 0
    summary = []
    for s in sentence_scores:
        if count >= limit:
            break
        else:
            include = True
            for ps in summary:
                sim = similarity(s[3], ps[3])
                if sim > sim_threshold:
                    include = False
            if include:
                summary.append(s)
                if limit_type == 'words':
                    count += len(s[1].split())
                elif limit_type == "bytes":
                    count += len(s[1])
    if reorder:
        summary = [s[1] for s in sorted(summary, key=lambda x: x[0])]
    else:
        summary = [s[1] for s in summary]
    return summary
