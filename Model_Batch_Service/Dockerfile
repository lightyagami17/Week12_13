# FROM jupyter/scipy-notebook

# #RUN mkdir my-model
# #ENV MODEL_DIR=/home/jovyan/my-model
# #ENV MODEL_FILE_LDA=clf_lda.joblib
# #ENV MODEL_FILE_NN=clf_nn.joblib

# RUN pip install joblib

# COPY train.csv ./train.csv
# COPY test.csv ./test.csv

# COPY train.py ./train.py
# COPY inference.py ./inference.py

# RUN python3 train.py
# Use a base image with Python pre-installed
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your application files into the container
COPY train.py .
COPY inference.py .
COPY train.csv .
COPY test.csv .

# Install necessary dependencies
RUN pip install pandas scikit-learn joblib

# Command to run the inference.py script
CMD ["python", "inference.py"]
