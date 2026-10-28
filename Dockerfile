# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code, scripts, tests, and configuration
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Install the project in editable mode
RUN pip install --no-cache-dir -e .

# Create results directory
RUN mkdir -p results

# Expose port for Web UI dashboard
EXPOSE 5000

# Default command runs the experiment
CMD ["python", "scripts/run_experiment.py"]
