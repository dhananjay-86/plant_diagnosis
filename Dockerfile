# Dockerfile for Plant Disease Diagnostics App

# Use a Python 3.11 base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
# --no-cache-dir: don't store the downloaded packages, to keep the image smaller
# We need build-essential and other libs for Pillow/numpy/TF compilation if from source
RUN apt-get update && apt-get install -y build-essential && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove build-essential

# Copy the rest of the application code
COPY . .

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable for the port
ENV PORT=8080

# Command to run the application using the production server
# Use -u for unbuffered output to see logs in real time
CMD ["python", "-u", "deploy.py"]
