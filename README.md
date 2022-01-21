# Ner Problem VSLP 2021
## Introduction
Named Entity Recognition (NER) is the task of tagging entities in text with their corresponding type, which is an important task in the field of Natural Language Processing. In this report, we used XLM-RoBERTa (XLM-R), which is a transformer-based multilingual masked language model pre-trained on text in 100 languages. We presented a simple approach to entity extraction, while integrating the model into an annotation engine called Doccano for automatic labeling. Besides, we also used LIME and AllenNLP to interpret the model's predictions. All results were evaluated on the VLSP 2021 dataset, which we customized to fit the purpose of the problem.

We trained the XLM-RoBERTa + Softmax model for the NER problem of the Vietnamese dataset with 15 types of entities. In general, our model gives good predictions in most cases and only poor in case of semantic ambiguity because of natural language features that are difficult for even humans to recognize. In addition, we also integrated the model into Doccano https://github.com/doccano/doccano to support the labeling task. Besides, we also applied XAI to our model using LIME.

## Features
### Processing Data
We have preprocessed input text  and postprocessed output thought some methods for improving performance predict. We have packed in (processing/post_processing.py and processing/pre_processing.py)
### Training
We have trained model and packed in (/modeling.py)
### Interpret by LIME methods
We have used LIME method for interpreting our model. We have packed in xai.py
### Integrate
We use FaskAPI for get output from model predicted. Can use API for Integrating into Docano or any applications be supported to anotation label for Ner Problem. You can use API in app.py
## Usage
Step 1: clone source code From source code:
'''
$ git clone https://github.com/ndtands/NER_VLSP2021.git

'''

Step 2: create virtual environment (ex: .env):
'''
$ python3 -m venv .env

'''
#### Mac OS / Linux
'''
source .env/bin/activate
'''
#### Windows
'''
.env\Scripts\activate
'''

Step 3: Install requirements.txt into environment have created before:
'''
pip install -r requirements.txt
'''

Step 4: After have intalled package, can run app.y:
'''
$ python app.py
'''

Step 5: Assess for addess 127.0.0 for using it.

## Conclude
We provide open source for anyone can use and deverlop it. If you have any contribute please push it.
If you have problem, you can comment in here or send problem to address:
1. Lengocloi1805@gmail.com (Lê Ngọc Lợi)
2. ngtinh98@gmail.com (Nguyễn Thị Tình)
3. ndtan.hcm@gmail.com (Nguyễn Duy Tân)
4. pvm26042000@gmail.com (Phạm Văn Mạnh)
