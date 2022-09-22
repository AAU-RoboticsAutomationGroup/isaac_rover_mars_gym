# General
This repo contains the implementation of ExoMy in Isaac Gym Preview 3 made by Group 666 (Spring 2022)

The setup specifically trains ExoMy to navigate in an unstructured Mars environment.

![space_rover](https://user-images.githubusercontent.com/10414639/191826045-a9c50712-2d81-4dd1-91fe-7e0dc1a5123d.png)

**Training**

https://user-images.githubusercontent.com/10414639/191824917-84412e61-6f6e-46e4-aaa1-2c7245a8e16c.mp4

**Testing**

https://user-images.githubusercontent.com/10414639/191824209-99b48773-32ee-4453-b3c9-88b8f5be6240.mp4


# Useful commands
[Commands](docs/Commands.md)

# Install isaac-rover-mars-gym

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
  
  1. Download Isaac Gym from https://developer.nvidia.com/isaac-gym
  2. Unzip Isaac Gym
  <!-- 3. nano isaacgym/docker/run.sh
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
```-->
3. nano isaacgym/docker/Dockerfile
4. Insert the follwing code at the bottom of the file and save.

```
RUN git clone https://github.com/AAU-RoboticsAutomationGroup/isaac_rover_mars_gym.git /home/gymuser/isaac_rover_mars_gym
RUN pip3 install -e /home/gymuser/isaac_rover_mars_gym/.
RUN git clone https://github.com/Toni-SM/skrl.git /home/gymuser/skrl
RUN pip3 install -e /home/gymuser/skrl/.
WORKDIR /home/gymuser/isaac_rover_mars_gym

```
<!---#7. sudo groupadd docker
#8. sudo gpasswd -a $USER docker
#9. restart PC-->
5. bash docker/build.sh
6. docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --network=host --gpus=all --name=isaacgym_exomy_container isaacgym /bin/bash
#11. bash docker/run.sh DISPLAYPORT
7. Enter container from different terminals --- sudo docker exec -it isaacgym_container bash 
  

<!---</details>--->

### 2. Run Isaac Gym
1. cd isaacgymenvs
2. python train.py


# RL Architecture

![rl_architecture](https://user-images.githubusercontent.com/10414639/191829922-0eb13701-f459-4f76-b972-9ac8f095d184.png)

![actor_network](https://user-images.githubusercontent.com/10414639/191830043-5c3ccf35-4297-4219-8d19-ce54e9d02a1e.png)


# Other questions
For other questions, contact 

- Anton: abmo19@student.aau.dk
- Emil: etpe19@student.aau.dk
- Jacob: jknuds19@student.aau.dk
