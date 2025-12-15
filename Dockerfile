FROM ubuntu:22.04

RUN apt-get update && apt-get install -y git \
                                         cmake \
                                         build-essential \
                                         wget \
                                         curl \
                                         libcurl4-openssl-dev

RUN git clone https://github.com/ggml-org/llama.cpp /opt/llama.cpp

WORKDIR /opt/llama.cpp
RUN mkdir build && cd build && cmake .. && cmake --build . --config Release
