FROM ubuntu:22.04

# kubernetes server requirements
RUN addgroup --gid 1000 group && \
  adduser --gid 1000 --uid 1000 --disabled-password --gecos User user

# GCC install
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get -y install gcc && \
    apt-get install -y libcurl4-openssl-dev && \
    apt-get install -y --no-install-recommends r-base && \
    apt-get clean

# install rpy2:
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
ENV CRAN_MIRROR=https://cloud.r-project.org \
    CRAN_MIRROR_TAG=-cran40
ARG RPY2_VERSION=RELEASE_3_5_6
ARG RPY2_CFFI_MODE=BOTH
COPY install_apt.sh /opt/
COPY install_rpacks.sh /opt/
COPY install_pip.sh /opt/
RUN \
  sh /opt/install_apt.sh \
  && python3 -m venv /opt/python3_env \
  && source /opt/python3_env/bin/activate \
  && sh /opt/install_rpacks.sh \
  && sh /opt/install_pip.sh \
  && rm -rf /tmp/* \
  && apt-get remove --purge -y $BUILDDEPS \
  && apt-get autoremove -y \
  && apt-get autoclean -y \
  && rm -rf /var/lib/apt/lists/*
RUN \
  source /opt/python3_env/bin/activate \
  && python3 -m pip --no-cache-dir install git+https://github.com/rpy2/rpy2.git@${RPY2_VERSION} \
  && rm -rf /root/.cache
#### rpy2 installed

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
RUN pip3 install rpy2

# Install DNAshapeR from Bioconductor
RUN R -e "install.packages('BiocManager', repos='http://cran.us.r-project.org')"
RUN R -e "BiocManager::install('DNAshapeR')"

# for testing
COPY small_unbalanced_test_dataset_randombg.fasta test.fasta
COPY src/bindmate bindmate
# RUN /src/venv/bin/python bindmate/install_dnashape.py

USER 1000
WORKDIR /src/