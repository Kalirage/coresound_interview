
# Use an official Python runtime as a parent image
FROM python:3.11.5

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Create a data directory
RUN mkdir data

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt