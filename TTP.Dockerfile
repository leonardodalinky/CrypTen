FROM anibali/pytorch:2.0.1-cuda11.8

# Install crypten
ENV PYTHONPATH=/framework
WORKDIR /framework
COPY crypten /framework/crypten
COPY examples /framework/examples
COPY configs /framework/configs
COPY requirements.txt /framework/requirements.txt
RUN pip install -r requirements.txt

# Scripts for TTP Server
WORKDIR /app
COPY scripts/ttp_server.py /app/ttp_server.py

ENTRYPOINT ["python", "ttp_server.py"]
