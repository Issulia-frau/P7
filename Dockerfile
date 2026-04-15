# Use official slim Python image
FROM python:3.11.9-slim

# Avoid Python buffering
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies for FAISS, PyTorch, and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest version
RUN python -m pip install --upgrade pip

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Expose the port for FastAPI
EXPOSE 8000

# Command to run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]