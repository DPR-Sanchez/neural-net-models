xhost +
docker run --net=host --env=DISPLAY -it --rm --volume $HOME/.Xauthority:/root/.Xauthority:rw neur
