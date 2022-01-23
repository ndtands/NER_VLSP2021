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
**Step 1**: Clone source code From source code:
```
$ git clone https://github.com/ndtands/NER_VLSP2021.git
```

**Step 2**: Download pretrain folder from huggingface or can use "xml-roberta-base" to dowload direct path from.

**Step 3**: Setup virtual environment (ex: .env):
```
Create environment
$ python3 -m venv .env or conda create --name .env

Mac OS / Linux activate:
$ .env/bin/activate

Windows activate:
$ .env\Scripts\activate

Conda activate:
$ Conda activate .env
```

**Step 4:** Install requirements.txt into environment have created before:
```
$ pip install -r requirements.txt
```

**Step 5:** Training Model:
For training model can follow our notebook: TEST.ipynb or you can dowload our trained weight file for data VLSP 2021.
'''
https://drive.google.com/file/d/1sPCnqT1m_tvj-V_9e6qucXck4ydiiWWJ/view?usp=sharing
'''

**Step 6**: Affer complete process training, you can runing command 'python app.py' and access addess '127.0.0:5000' for using application.

## Contribution & Contact
We provide open source for anyone can use and develop it. If you have any contributions, please push it.
If you have any problems, you can comment here or send problems to address:
1. lengocloi1805@gmail.com (Lê Ngọc Lợi)
2. ngtinh98@gmail.com (Nguyễn Thị Tình)
3. ndtan.hcm@gmail.com (Nguyễn Duy Tân)
4. pvm26042000@gmail.com (Phạm Văn Mạnh)
