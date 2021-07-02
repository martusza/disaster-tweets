FROM python:3.8.2-alpine3.11

COPY ./src/app ./src/app

RUN pip install Flask

CMD ["python", "./src/app/flask_api_test.py"]