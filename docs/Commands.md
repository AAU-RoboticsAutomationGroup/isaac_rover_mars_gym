# General
- To edit files in docker, use **Remote - Containers** extension for visual studio code. Connect to docker through the *Remote Explorer* tab.

# Commands for docker
 - **docker ps**
 Shows currently active docker containers

- **docker start *some_container***
Starts *some_container*. For the isaac gym container: *docker start isaacgym_container *

- **docker stop *some_container***
Stops *some_container*. For the isaac gym container: *docker stop isaacgym_container *

- **sudo docker exec -it isaacgym_container bash**
Access the active docker *isaacgym_container* through a terminal

# Commands for Isaac
To run the standard exomy environment:
1. (In the active isaacgym container) ***cd envs/***
2. ***python exomy.py***

To run our exomy environment:
1. (In the active isaacgym container) ***cd RL/***
2. ***python train.py task=Exomy***
- **IF ANY PACKAGE IS FAILING TO IMPORT THAT WE HAVE DOWNLOADED***
1. GO INTO CORRESPONDING FOLDER AT DO:
2. **pip install -e .**

To run Issac gym examples, clone them to docker.
Upon execution of train.py, if error: ModuleNotFoundError: No module named *hydra*
1. **cd IsaacGymEnvs**
2. **pip install -e .**

# Other commands
To run docker, such that programs can be installed through command-line
1. **sudo docker exec -u root -it isaacgym_container /bin/bash**

# Visual Studio Code and Live Share
If live share wont work with the docker container, install code in the docker:
1. Access docker in root: **sudo docker exec -it -u root isaacgym_container /bin/bash**
2. Update the packages index and install the dependencies by running the following command as a user with sudo privileges: **apt update**
3. **apt install software-properties-common apt-transport-https wget**
4. Import the Microsoft GPG key using the following wget command: **wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | apt-key add -**
5. And enable the Visual Studio Code repository by typing: **add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"**
6. Once the apt repository is enabled , install the Visual Studio Code package: **apt install code**
