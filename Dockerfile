FROM python:3.7

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
COPY . .
COPY neupy/plots.py /usr/local/lib/python3.7/site-packages/neupy/algorithms/plots.py
COPY neupy/base.py /usr/local/lib/python3.7/site-packages/neupy/algorithms/base.py

CMD ["python", "GUI.py"]
