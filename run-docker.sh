docker build -t pytorch .
nvidia-docker run -v .:/workspace --name=pytorch-test --rm -it pytorch /bin/bash
