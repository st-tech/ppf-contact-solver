FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV NVIDIA_DRIVER_CAPABILITIES=utility,compute
ENV LANG=en_US.UTF-8

RUN apt update
RUN apt install -y git python3 curl

RUN mkdir -p /root/ppf-contact-solver
WORKDIR /root/ppf-contact-solver

RUN curl https://raw.githubusercontent.com/st-tech/ppf-contact-solver/refs/heads/main/warmup.py -o warmup.py
RUN python3 warmup.py

WORKDIR /root/
RUN rm -rf ppf-contact-solver
RUN rm -rf /var/lib/apt/lists/*