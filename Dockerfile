# Use Python 3.11 slim image for better performance
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code - EXCLUDE problematic directories
COPY . .

# CRITICAL: Remove analysis cache and other problematic directories that cause deployment failures
RUN rm -rf .analysis_cache/ \
    && rm -rf .rag_cache/ \
    && rm -rf __pycache__/ \
    && rm -rf .git/ \
    && rm -rf .vscode/ \
    && rm -rf .idea/ \
    && rm -rf *.pkl \
    && rm -rf temp/ \
    && rm -rf tmp/ \
    && find . -name "*.pyc" -delete \
    && find . -name "__pycache__" -type d -exec rm -rf {} + || true

# Create necessary runtime directories (empty)
RUN mkdir -p .analysis_cache/analyses \
    && mkdir -p .analysis_cache/embeddings

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose the port that the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "api_backend.py"]