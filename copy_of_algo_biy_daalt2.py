from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok

#sanguud
"""

import pickle
import pandas as pd
import re
import os
import numpy as np
import gensim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim.models import Doc2Vec
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import hunspell
from spellchecker import SpellChecker

from google.colab import drive
drive.mount('/content/drive')

"""#datas"""

train = pd.read_csv("/content/drive/My Drive/algo biy daalt/train_data.csv", encoding='utf-8')

"""#uguulber tanilt"""

train = train[['medee', 'turul']]
train.head().T
train.groupby('turul').describe()

train.head()

train.turul.unique()

train['NUM_type_text']=train.turul.map({'байгал орчин':0,'боловсрол':1,'спорт':2,'технологи':3,'улс төр':4,'урлаг соёл':5,'хууль':6,'эдийн засаг':7,'эрүүл мэнд':8})
train.head()
x_train, x_test, y_train, y_test = train_test_split(train.medee, train.NUM_type_text, random_state=0, test_size=0.2)

"""Жишээ: Манай багш мундаг гэвэл "Манай багш" , "багш мундаг"
"""

#ugsiig too bolgoh ngram_range ni 2 2 ugeer bagtsalj too bolgoj bga
vect = CountVectorizer(ngram_range=(2,2))
#surgaltiin datag toon vectorluu horvuuleh
X_train = vect.fit_transform(x_train)
X_test = vect.transform(x_test)

#surgalt
#tulhuur ug Naive Bayes tgd sanana biz zalhuu hurle ahha yr ni bol dawtamjind suurilj text angildag model
mnb = MultinomialNB(alpha =0.2)
mnb.fit(X_train,y_train)
result= mnb.predict(X_test)
#model accuracy
print(accuracy_score(result,y_test))

#garaltiin utga
def predict_news(news):
    test = vect.transform([news])
    pred= mnb.predict(test)
    if pred  == 0:
         return 'байгал орчин'
    elif pred ==1 :
        return 'боловсрол'
    elif pred ==2 :
        return 'спорт'
    elif pred ==3 :
        return 'технологи'
    elif pred ==4 :
        return 'улс төр'
    elif pred ==5 :
        return 'урлаг соёл'
    elif pred ==6 :
        return 'хууль'
    elif pred ==7 :
        return 'эдийн засаг'
    else :
        return 'эрүүл мэнд'

r = predict_news(final_sentence)
print (r)

# Үг шалгах функц
def ug_shalgah(test):
    ugnuud = test.replace(',', '').replace('.', '').split()

    zuw_ug = []
    buruu_ug = []

    for ug in ugnuud:
        if hspell.spell(ug) or ug in eng_spell:
            zuw_ug.append(ug)
        else:
            buruu_ug.append(ug)

    return zuw_ug, buruu_ug

# Үндэс үг олох функц
def undes_ug(zuw_ug):
    stems = {}
    for ug in zuw_ug:
        stem = hspell.stem(ug)
        if stem:
            stems[ug] = [s.decode('utf-8') for s in stem]
        else:
            stems[ug] = None

    return stems

# Буруу үгсэд санал болгох үгс
def buruu_ug_sanal(buruu_ug):
    suggestions = {}
    for ug in buruu_ug:
        suggestions[ug] = hspell.suggest(ug)
    return suggestions

# Медээний үндэс үг олох функц
def process_medee(medee_text):
    # Зөв болон буруу үгнүүдийг шалгах
    zuw_ug, buruu_ug = ug_shalgah(medee_text)
    # Зөв үгнүүдийн үндэс үгийг олох
    stems = undes_ug(zuw_ug)
    # Үндэс үгийг өгүүлбэр хэлбэрээр буцаах
    processed_text = " ".join([", ".join(stem_list) if stem_list else word for word, stem_list in stems.items()])
    return processed_text

# DataFrame-ийн medee баганыг боловсруулах
train['medee'] = train['medee'].apply(process_medee)

# Үр дүнг харах
# train

# Серверийн код
app = Flask(__name__)
CORS(app, resources={r"/process": {"origins": "*"}})

public_url = ngrok.connect(5000)
print('Public URL:', public_url)

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()

    if 'test' not in data:
        return jsonify({"error": "Missing 'test' parameter"}), 400

    test = data['test']

    try:
        zuw_ug, buruu_ug, ugnuud = ug_shalgah(test)
        stems = undes_ug(ugnuud)
        final_sentence = undes_ug_sentence(stems)
        suggestions = buruu_ug_sanal(buruu_ug)

        sedev = predict_news(test)

        response = {
            "zuw_ug": zuw_ug,
            "buruu_ug": buruu_ug,
            "final_sentence": final_sentence,
            "suggestions": suggestions,
            "sedev": sedev
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000) 