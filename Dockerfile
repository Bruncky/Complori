# tensorflow base-images are optimized: lighter than python-buster + pip install tensorflow
FROM tensorflow/tensorflow:2.10.0

# OR for Apple Silicon, use this base image instead
# FROM armswdev/tensorflow-arm-neoverse:r22.09-tf-2.10.0-eigen

# We strip the requirements from useless packages like `ipykernel`, `matplotlib` etc...
COPY requirements.txt requirements.txt

RUN pip install -U pip cython wheel
RUN pip install -r requirements.txt

COPY complori complori
# Normally, I would have data on GCS or S3
COPY data data
COPY setup.py setup.py

RUN pip install .

CMD uvicorn complori.api:app --host 0.0.0.0 --port $PORT
