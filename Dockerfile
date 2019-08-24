FROM python:3.7

COPY . .
RUN pip install pysimplegui neupy dill sklearn tensorflow numpy pandas

CMD ["python", "GUI.py"]
