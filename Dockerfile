FROM pytorch/pytorch:latest
RUN apt-get -y -qq update && \
    pip install numpy matplotlib librosa
COPY . .