FROM kaixhin/cuda-caffe:8.0

RUN apt-get update && apt-get install -y \
  curl \
  build-essential
# update pip
RUN curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
RUN python get-pip.py; hash -r


# install ipython & other python packages
RUN pip install \
  nose \
  numpy \
  scipy \
  Pillow \
  ai-integration==1.0.11

# create directory to house model, code & images
RUN mkdir /root/caffe/deepdream

RUN ./scripts/download_model_binary.py models/bvlc_googlenet/

# Fix for boto3 conflicting with installed urllib3
RUN pip install --ignore-installed urllib3==1.24

COPY model_src model_src
WORKDIR model_src

CMD []
ENTRYPOINT ["python", "entrypoint.py"]
