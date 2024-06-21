# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory in the container
# WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Copy .env file to the container
COPY .env .env

# Expose the port the app runs on
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Ensure the .env variables are available in the container
RUN export $(cat .env | xargs)

# Command to run the application
CMD ["python", "app.py"]
