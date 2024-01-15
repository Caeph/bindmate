FROM ubuntu:22.04

# kubernetes server requirements
RUN addgroup --gid 1000 group && \
  adduser --gid 1000 --uid 1000 --disabled-password --gecos User user

# get java and python
RUN apt-get update && apt-get install -y --fix-missing python3 && \
    apt-get install -y python3.10-venv
RUN apt install -y openjdk-11-jdk && apt install -y openjdk-11-jre

# venv, prep venv
RUN python3 -m venv /src/venv
ENV PATH=/src/venv/bin:$PATH
WORKDIR /src
COPY  requirements.txt .
RUN . /src/venv/bin/activate && pip3 install wheel && pip3 install -r  requirements.txt

# for testing
COPY small_unbalanced_test_dataset_randombg.fasta test.fasta
COPY src/bindmate bindmate
