# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose the port used in your app
EXPOSE 10000

# Run the app
CMD ["python", "flask_web_app.py"]
