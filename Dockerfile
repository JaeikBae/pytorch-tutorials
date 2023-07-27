FROM pytorch/pytorch:latest
RUN apt-get -y -qq update
RUN apt-get install -y python3-tk
RUN pip install numpy matplotlib librosa PySide6
RUN pip install torchsummary