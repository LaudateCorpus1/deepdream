# Deep Dream

<p>
    <a href="https://cloud.docker.com/u/deepaiorg/repository/docker/deepaiorg/deepdream">
        <img src='https://img.shields.io/docker/cloud/automated/deepaiorg/deepdream.svg?style=plastic' />
        <img src='https://img.shields.io/docker/cloud/build/deepaiorg/deepdream.svg' />
    </a>
</p>

This model has been integrated with [ai_integration](https://github.com/deepai-org/ai_integration/blob/master/README.md) for seamless portability across hosting providers.

# Overview

Implementation of Deep Dream.

Nvidia-Docker is required to run this image.

# For details see [Deep Dream](https://deepai.org/machine-learning-model/deepdream) on [Deep AI](https://deepai.org)

# Quick Start

docker pull deepaiorg/deepdream

### HTTP
```bash
nvidia-docker run --rm -it -e MODE=http -p 5000:5000 deepaiorg/deepdream
```
Open your browser to localhost:5000 (or the correct IP address)

### Command Line

Save your image as content.jpg in the current directory.
```bash
nvidia-docker run --rm -it -v `pwd`:/shared -e MODE=command_line deepaiorg/deepdream --image /shared/content.jpg --output /shared/output.jpg
```
# Docker build
```bash
docker build -t deepdream .
```

# Author Credit

Original concept ipython notebook:

* [Alexander Mordvintsev](mailto:moralex@google.com)
* [Michael Tyka](https://www.twitter.com/mtyka)
* [Christopher Olah](mailto:colah@google.com)

Loosely based on this implementation:

https://github.com/graphific/DeepDreamVideo

Simplification, streamlining, and updating to work in the year 2019 by DeepAI.
