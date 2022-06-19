from solver import NerSolver
from flask import Flask,render_template,request, jsonify
from utils import config


ner = NerSolver(config)
app = Flask(__name__,   static_url_path='',  static_folder='static')

@app.route('/predict')
def predict():
	return render_template('predict.html')


@app.route('/predict', methods =["POST"])
def api_predict():
    text = request.json['text']
    rs = ner.solve(text)
    out = {'rs': rs}
    return jsonify(out)

        

@app.route('/')
def index():
	return render_template('interpret.html', raw_text = '', result='')

if __name__ == '__main__':
    app.run(host = config.host, port=config.port, debug= config.is_prod)