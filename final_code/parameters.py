parameters = {"text_folder": "../data/test_files", # folder with text file/s
              "topic_threshold": 0.1,
              "sim_threshold": 0.9,
              "n_top" :3,
              "remove_stopwords": True,
              "remove_punct": True,
              "language": "english", # "german"
              "topic": "news", # "general"
              "tfidf_path": "../data",
              "refdoc_path": "../data/ref_docs_clean.txt",
              "load_tfidf_model": True,
              "save_tfidf_model": True,
              "limit_type": "bytes", #"words"
              "limit": 665,
              "reorder": True}
