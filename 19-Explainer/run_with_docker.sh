docker build -t shap_test . 
docker run -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY="unix:0" -it shap_test bash

