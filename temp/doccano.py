from flask import Flask,url_for,render_template,request
import spacy
from spacy import displacy
import json
import torch
import time
from modelling import *
import utils

def convert2doccano(arr):
    if len(arr) < 1:
        return None
    text = ' '.join([i for i, j in arr])
    pos = 0
    start_end_labels = []
    for word, tag in arr:
        if len(start_end_labels) > 0 and tag == start_end_labels[-1][2]:
            temp = [start_end_labels[-1][0], pos+len(word), tag]
            start_end_labels[-1] = temp.copy()
        else:
            temp = [pos, pos+len(word), tag]
            start_end_labels.append(temp)
        pos += len(word) + 1
        
    ex = [{'start_offset': x[0], 'end_offset': x[1], 'label': x[2]} for x in start_end_labels if x[2]!= 'O']
    return ex

def loading(PATH = 'model/xlmr_span0_old.pt'):
    MAX_LEN = 256
    BS = 64
    DROPOUT_OUT = 0.4
    model_name = 'xlmr_softmax'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print(device, PATH)
    tag_values = ['PAD','ADDRESS','SKILL','EMAIL','PERSON','PHONENUMBER','MISCELLANEOUS','QUANTITY','PERSONTYPE',
            'ORGANIZATION','PRODUCT','IP','LOCATION','O','DATETIME','EVENT', 'URL']  
    with open('path.json', 'r', encoding= 'utf-8') as f:
        dict_path = json.load(f)
    dict_path = dict_path[model_name.split('_')[0]]
    dict_path['weight'] = PATH
    start = time.time()
    print("1. Loading some package")
    ner = NER(dict_path = dict_path, model_name = model_name, tag_value = tag_values , dropout = DROPOUT_OUT, max_len = MAX_LEN, batch_size = BS, device = device)
    print(f"===== Done !!! =====Time: {time.time() -start:.4} s =========")
    print('2.Load model')
    start = time.time()
    ner.model = load_model(ner.model,dict_path['weight'],device)
    print(f"===== Done !!! =====Time: {time.time() -start:.4} s =========")
    return ner

ner = loading('model/xlmr_span0_30t12.pt')

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

from flask import Flask, request, jsonify, Blueprint
from flask_restful import reqparse, abort, Api, Resource, marshal_with
import uuid
app = Flask(__name__)
api_bp = Blueprint('api', __name__)
api = Api(api_bp)

parser = reqparse.RequestParser(bundle_errors = True)
parser.add_argument('text', type=str, required=True, help='Content cannot be left blank')

class TextClassification(Resource):
    def post(self):
        data = {}
        try:
            args = parser.parse_args()
            text =  args.text
            res = ner.predict(text)
            data = convert2doccano(res)
        except Exception as e:
            data = {'error': 200}
        return data


api.add_resource(TextClassification, '/api/')
app.register_blueprint(api_bp)

if __name__=='__main__':
    app.run(host = '0.0.0.0', debug=True, port='3002', use_reloader=False)
