# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements-online-dyna.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV GOOGLE_APPLICATION_CREDENTIALS ./keys/datastore-key.json

# Run app.py when the container launches
#CMD ["gunicorn", "-b", "0.0.0.0:8080", "wsgi:app"]
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--worker-class", "gevent", "wsgi:app"]
