FROM python:3.9
LABEL authors="NikaShishkova"

WORKDIR /home/mma-parser
COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "./main.py"]
