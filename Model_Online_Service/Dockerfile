# Use a base image with Python 3.10.12 pre-installed
FROM python:3.10.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your application files into the container
COPY inference.py .
COPY test.csv .
COPY Lda.sav .
COPY NN.sav .
COPY svm.joblib .

# Install necessary dependencies
RUN pip install --upgrade pip
RUN pip install pandas scikit-learn joblib

# Command to run the inference.py script
CMD ["python", "inference.py"]
