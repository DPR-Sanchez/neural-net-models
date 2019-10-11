FROM tensorflow/tensorflow:1.15.0rc2-gpu-py3
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends python3-tk git

RUN git clone https://github.com/itdxer/neupy.git
WORKDIR /neupy
COPY neupy_override.txt requirements/main.txt 
RUN python setup.py install

WORKDIR /
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "GUI.py"]
