- Create dockerfile inside a folder
- Dockerfile:
    - From a built imgs
    - Run: during img creating (RUN apt-get update)
    - CMD: duiring container creation
    - entrypoint ?

- docker build -t dockerimg:1.0.0 . 
    - if other folder change "." to other folder name that have dockerfile

- Then we can run docker images to check the image info
- Then docker run to start the container (docker run -p port:port --name test -v folder/folder dockerimg:1.0.0 /bin/bash)...

- reference: https://github.com/wsargent/docker-cheat-sheet#dockerfile