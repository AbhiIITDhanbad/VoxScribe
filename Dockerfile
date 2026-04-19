FROM python:3.12-slim

RUN apt update -y && pip3 --no-cache-dir install --upgrade awscli

WORKDIR /app

COPY  . /app

RUN pip install -r requirements.txt

CMD ["python3", "fastapi_app.py"]

# docker run -d -e AWS_ACCESS_KEY_ID = "" -e AWS_SECRET_ACCESS_KEY = "" -p 8080:8080 speech-to-text