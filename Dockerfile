FROM ubuntu:lunar
RUN apt update && apt install -y \
    build-essential \
    curl \
    git \
    gcc-13 \
    g++-13 \
    wget \
    cmake \
    libprotobuf-dev \
    libgtest-dev \
    libboost-all-dev \
    libeigen3-dev \
    python3-dev \
    virtualenv

ENV TF_FILENAME libtensorflow-cpu-linux-x86_64-2.15.0.tar.gz
RUN wget -q --no-check-certificate https://storage.googleapis.com/tensorflow/libtensorflow/${TF_FILENAME} &&\
    tar -C /usr/local -xzf ${TF_FILENAME} &&\
    ldconfig /usr/local/lib

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
