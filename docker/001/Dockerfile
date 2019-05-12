FROM ubuntu:18.04

LABEL name="aiokinawa/traffic_signs"
LABEL version="001"

# Install anaconda
RUN apt update && apt install -y wget \
    && wget https://repo.continuum.io/archive/Anaconda3-5.3.0-Linux-x86_64.sh \
    && bash ./Anaconda3-5.3.0-Linux-x86_64.sh -b -p /root/anaconda3

# Add anaconda to PATH
ENV PATH="/root/anaconda3/bin:${PATH}"

# Update conda environment
COPY ./environment_linux.yaml /tmp/environment_linux.yaml
RUN conda env update -f /tmp/environment_linux.yaml
