# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install Poetry
RUN pip install --upgrade pip && \
    pip install --no-cache-dir poetry

# Install any needed packages specified in pyproject.toml
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --no-dev