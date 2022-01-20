from flask import Flask,url_for,render_template,request, jsonify
import json
import time
from modelling import *
from tqdm import tqdm, trange

from sklearn.model_selection import train_test_split
from processing.pre_processing import preprocessing_text

from xai import xai_lime
from define_name import *

def loading(config, PATH = 'model/xlmr_span1.pt'):
    start = time.time()
    print("1. Loading some package")
    ner = NER(config)
    print(f"===== Done !!! =====Time: {time.time() -start:.4} s =========")
    print('2.Load model')
    start = time.time()
    ner.load_model(PATH)
    print(f"===== Done !!! =====Time: {time.time() -start:.4} s =========")
    return ner

with open(PATH_CONFIG, 'r', encoding= 'utf-8') as f:
    config = json.load(f)
ner = loading(config, PATH_MODEL)

from flaskext.markdown import Markdown

app = Flask(__name__,   static_url_path='', 
            static_folder='static')
Markdown(app)


@app.route('/interpret')
def interpret():
	return render_template('interpret.html')

@app.route('/predict')
def predict():
	return render_template('predict.html')


@app.route('/predict', methods =["POST"])
def api_predict():
    text = request.json['text']
    rs = ner.predict(text)
    out = {'rs': rs}
    return jsonify(out)

        

@app.route('/')
def index():
	return render_template('interpret.html', raw_text = '', result='')


@app.route('/interpret',methods=["POST"])
def api_interpret():
    # try:
    print("Server received data: {}, {}".format(request.json['text'], request.json['idx']))
    print("Server received data: {}, {}".format(type(request.json['text']), type(request.json['idx'])))
    text = request.json['text']
    id_word = int(request.json['idx'])

    rs = xai_lime(ner, text, id_word)
    return jsonify(rs)
    # except:
    #     return jsonify({
    #         "message": "error in server"})



if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False)

