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
2. ***python train2.py task=Exomy***


To run Issac gym examples, clone them to docker.
Upon execution of train.py, if error: ModuleNotFoundError: No module named *hydra*
1. **cd IsaacGymEnvs**
2. **pip install -e .**

# Other commands
To run docker, such that programs can be installed through command-line
1. **sudo docker exec -u root -it isaacgym_container /bin/bash**
