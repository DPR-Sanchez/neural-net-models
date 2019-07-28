FROM python:3.7

COPY . .
RUN pip install pysimplegui neupy dill sklearn tensorflow numpy

CMD ["python", "GUI.py"]