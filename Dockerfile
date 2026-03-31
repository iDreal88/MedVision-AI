# Final Production Build for Master Thesis - MedVision-Agent v2
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# Install standard system dependencies for OpenCV and MedVision
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the standard Hugging Face Space port
EXPOSE 7860

# Run the application
CMD uvicorn api.main:app --host 0.0.0.0 --port 7860 --proxy-headers
