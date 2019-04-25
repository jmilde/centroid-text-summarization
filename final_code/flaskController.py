#!/usr/bin/python3
from flask import Flask, flash, request, redirect, url_for
from flask_cors import CORS
import os
import shutil
from werkzeug.utils import secure_filename
import zipfile

import summarize

"""

then to start run:
export FLASK_APP=main.py
export FLASK_DEBUG=1 (optional, to reload upon changes automatically)
python -m flask run

example calls:

curl -X GET localhost:5000/welcome

curl -F 'inputzip=@testfiles.zip' -X POST localhost:5000/summarize



"""


app = Flask(__name__)
app.secret_key = "super secret key"
CORS(app)

TEMP_DATA_FOLDER = os.path.abspath(os.path.join(os.getcwd(), '../temp_input'))


@app.route('/welcome', methods=['GET'])
def dummy():
    return "Hello stranger, can you tell us where you've been?\nMore importantly, how ever did you come to be here?\n"




##################### Summarization #####################
@app.route('/summarize', methods=['POST'])
def getsummary():

    if request.method == 'POST':
        if 'inputzip' not in request.files:
            flash('No zip file specified')
            return redirect(request.url)

        # TODO: write check if zip does not contain subfolders
        
        _zipfile = request.files['inputzip']
        # unzip to data folder
        if os.path.exists(TEMP_DATA_FOLDER):
            shutil.rmtree(TEMP_DATA_FOLDER)
        os.makedirs(TEMP_DATA_FOLDER)
            
        zip_ref = zipfile.ZipFile(_zipfile, 'r')
        zip_ref.extractall(TEMP_DATA_FOLDER)
        zip_ref.close()
        #for key, val in parameters.parameters.items():
            #exec(key+"=val")

        out_path = '../data/test_summary.txt'
        topic_threshold = 0.1
        sim_threshold= 0.9
        n_top=3
        remove_stopwords= True
        remove_punct= True
        language= "english"
        topic= "news"
        tfidf_path= "../data"
        refdoc_path= "../data/ref_docs_clean.txt"
        load_tfidf_model= True#False # set to False if first run
        save_tfidf_model= True
        limit_type= "bytes"
        limit= 665
        reorder= True

        
        summary = summarize.summarize(TEMP_DATA_FOLDER, out_path, topic_threshold, sim_threshold, n_top, remove_stopwords, remove_punct, language, topic, tfidf_path, refdoc_path, load_tfidf_model, save_tfidf_model, limit_type, limit, reorder)

        return '\n'.join(summary) # perhaps joining on whitespace is better
        
    else:
        return 'METHOD %s not allowed.\n' % request.method    



    
if __name__ == '__main__':


    
    port = int(os.environ.get('PORT',5000))
    app.run(host='localhost', port=port, debug=True)
