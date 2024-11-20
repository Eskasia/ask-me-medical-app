# Use a specific version of Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables for Python (optional but recommended)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file to leverage Docker cache
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose port 5000
EXPOSE 5000

# Default command to run the application
CMD ["python", "app.py"]
