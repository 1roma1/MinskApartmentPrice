FROM python:3.8-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY server/server.py .
COPY server/run.sh .
COPY model/model.pk .

RUN chmod +x run.sh

CMD ["run.sh"]