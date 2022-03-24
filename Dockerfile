FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY server.py .
COPY model/model.pk .

EXPOSE 8080

CMD ["python", "server.py"]