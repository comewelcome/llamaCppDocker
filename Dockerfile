FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    pciutils \
    build-essential \
    cmake \
    curl \
    libcurl4-openssl-dev \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone -b mtp-clean https://github.com/am17an/llama.cpp.git

RUN cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DGGML_CURL=ON
RUN cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-server
RUN cp llama.cpp/build/bin/llama-* llama.cpp/

# Create the modele directory in the container
RUN mkdir -p /app/modele

# llama.cpp uses LLAMA_CACHE environment variable to decide where to store models
ARG LLAMA_CACHE=/app/modele
ENV LLAMA_CACHE=${LLAMA_CACHE}
ENV XDG_CACHE_HOME=${LLAMA_CACHE}

ENTRYPOINT ["./llama.cpp/llama-server"]
