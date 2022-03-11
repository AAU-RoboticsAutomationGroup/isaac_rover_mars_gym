# isaac-rover

<!---<details><summary>Docker (click to expand)</summary>--->

### Requirements

- Ubuntu 18.04, or 20.04.
- Python 3.6, 3.7, or 3.8
- Minimum recommended NVIDIA driver version: 470.74 (470 or above required)

  
### Dependencies
  ```bash
# Docker
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
  
# Setting up NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
# Install the nvidia-docker2 package (and dependencies) after updating the package listing:
sudo apt-get update
sudo apt-get install -y nvidia-docker2
# Restart the Docker daemon to complete the installation after setting the default runtime:
sudo systemctl restart docker

```
  
### 1. Download Isaac Gym
  
  1. Download Isaac Gym from https://forums.developer.nvidia.com/t/isaac-gym-preview-3-release-now-available/193865
  2. Unzip Isaac Gym
  3. nano isaacgym/docker/run.sh
  4. Remove all text and paste
```
#!/bin/bash
set -e
set -u

if [ $# -eq 0 ]
then
    echo "running docker without display"
    docker run -it --network=host --gpus=all --name=isaacgym_container isaacgym /bin/bash
else
    export DISPLAY=$DISPLAY
	echo "setting display to $DISPLAY"
	xhost +
	docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --network=host --gpus=all --name=isaacgym_container isaacgym /bin/bash
	xhost -
fi
```
5. nano isaacgym/docker/Dockerfile
6. Insert the follwing code at the bottom of the file and save.

```
RUN git clone https://github.com/ExoMyRL/isaac_rover.git /home/gymuser/isaac_rover
WORKDIR /home/gymuser/isaac_rover
```
7. bash docker/build.sh
8. bash docker/run.sh <display>
9. Enter container from different terminals --- sudo docker exec -it isaacgym_container bash 
  

<!---</details>--->
