FROM pytorch/pytorch:latest
RUN apt-get -y -qq update && \
    pip install numpy matplotlib librosa PySide6
RUN apt-get install -y python3-tk