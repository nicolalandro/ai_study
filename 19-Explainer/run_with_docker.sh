docker build -t shap_test .
# sudo apt-get install x11-xserver-utils
xhost + 
docker run -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -it shap_test bash

