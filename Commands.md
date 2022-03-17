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
To run the exomy environment:
1. (In the active isaacgym container) ***cd envs/***
2. ***python exomy.py***

To run Issac gym examples, clone them to docker.
Upon execution of train.py, if error: ModuleNotFoundError: No module named *hydra*
1. **cd IsaacGymEnvs**
2. **pip install -e .**
