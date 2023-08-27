# Week12_13
Complete code for Week 12-13 Project
Install Python if not already installed. I've used the Python 3.10.12 version
To run this application, clone the code into your machine, go to the Model_Online_Service directory, and run the below command

python inference.py

In order to create and run the container, follow the below steps
1. Go to the Model_Online_Service directory
2. Run the below command. You can give whatever image name you want
    docker build -t <nameofimage> .
3. After building, you can check whether the image is built or not by using the below command
    docker images
4. Now once the image is built successfully, you have to create and run the container by using the below command
    docker run -d -v $(pwd):/app --name <nameofimage> <nameofcontainer>

    Here, you can provide any container name
    pwd = Linux command to provide the current working directory
    /app = Our directory where the app will run. This is mentioned and can be changed in the Dockerfile.
5. In order to check whether our container is up and running, we can use the below command
    docker ps
6. In order to view the containers that are either stopped or exited, you can use the below command
    docker ps -a
