# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set environment variables to ensure Python outputs everything to the terminal and to not generate .pyc files
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app/

# Install system dependencies for Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Install project dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose port 8000 for the application
EXPOSE 8000

# Command to run the application using gunicorn
CMD ["gunicorn", "-w", "4", "--bind", "0.0.0.0:8000", "main:app"]