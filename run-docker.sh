docker build -t pytorch .
xhost +
nvidia-docker run \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v .:/workspace \
    --rm -it \
    pytorch \
    /bin/bash
