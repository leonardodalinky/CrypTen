FROM anibali/pytorch:2.0.1-cuda11.8

# Install crypten
ENV PYTHONPATH=/framework
WORKDIR /framework
COPY crypten /framework/crypten
COPY examples /framework/examples
# COPY configs /framework/configs
COPY requirements.txt /framework/requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app
# Expect /app to be mounted as a volume

ENTRYPOINT ["python", "launcher.py"]
