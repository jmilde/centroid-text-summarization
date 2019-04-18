from util import  load_embedding, clean_txt, topic_words, weight_sentences, get_centroid, select_ntop, score_sentences, select_sentences
from nltk.corpus import stopwords
import string

def summarize(topic_threshold,
              sim_threshold,
              n_top,
              remove_stopwords,
              remove_punct,
              language,
              topic,
              tfidf_path,
              refdoc_path,
              load_tfidf_model,
              save_tfidf_model,
              limit_type,
              limit,
              reorder)


    ### LOAD EMBEDDING MODEL
    model = load_embedding(language, topic)

    # load data
    txt = [open(text_folder+ "/" + f).read() for f in os.listdir(text_folder)]
    ### Preprocess the data
    plain_txt = " ".join(txt)

    # remove stopwords and punctuation
    remove = set()
    if remove_stopwords:
        remove.update(stopwords.words(language))
    if remove_punct:
        remove.update(list(string.punctuation))

    clean_txt, raw_sents = clean_txt(plain_txt, remove)

    # GET TOPIC WORDS
    centroid_words_weights, tfidf_scores, feature_names = topic_words(
        clean_txt, tfidf_path, topic_threshold, load=load_tfidf_model, refdoc_path=refdoc_path, save=save_tfidf_model)

    # weight sentences
    scores = weight_sentences(txt, centroid_words_weights, tfidf_scores, feature_names, remove)

    # if multidocument, select only top sentences
    if len(txt)>1:
        clean_sents, raw_sents = select_ntop(txt, scores, n_top, remove)


    # get centroid words
    centroid_words = list(centroid_words_weights.keys())
    centroid_vector = get_centroid(centroid_words, model)

    # score sentences
    sentence_scores = score_sentences(clean_sents, raw_sents, model, centroid_vector)

    # select sentences
    summary = select_sentences(sentence_scores, sim_threshold, limit_type, limit, reorder)
    return summary

if __name__ == "__main__":
    for key, val in params.items():
        exec(key+"=val")

    summarize(topic_threshold, sim_threshold, n_top, remove_stopwords, remove_punct, language, topic, tfidf_path, refdoc_path, load_tfidf_model, save_tfidf_model, limit_type, limit, reorder)
