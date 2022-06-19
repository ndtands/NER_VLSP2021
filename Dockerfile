FROM python:3.8.6

WORKDIR /ner_penhouse601
COPY . /ner_penhouse601

RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt

RUN chmod +x run.sh
CMD ./run.sh