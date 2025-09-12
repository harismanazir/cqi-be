# Multi-stage build to reduce memory usage
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies including git
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Upgrade pip and install wheel for binary packages
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install dependencies in stages to manage memory usage
# Stage 1: Install lighter dependencies first
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    flask>=2.3.0 \
    pydantic>=2.0.0 \
    python-multipart>=0.0.6 \
    python-dotenv>=1.0.0 \
    gunicorn>=21.2.0 \
    typing-extensions>=4.0.0

# Stage 2: Install Git packages
RUN pip install --no-cache-dir \
    GitPython==3.1.40 \
    PyGithub==1.59.1

# Stage 3: Install heavy ML packages one by one with memory optimization
RUN pip install --no-cache-dir --no-deps numpy>=1.24.0
RUN pip install --no-cache-dir scikit-learn>=1.3.0
RUN pip install --no-cache-dir langchain-core>=0.1.0
RUN pip install --no-cache-dir langchain-groq>=0.1.0
RUN pip install --no-cache-dir langgraph>=0.1.0

# Stage 4: Install sentence-transformers last (heaviest package)
# Use --find-links to prefer pre-built wheels
RUN pip install --no-cache-dir \
    --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
    sentence-transformers>=2.2.0

# Verify critical imports work
RUN python -c "import fastapi, uvicorn, git, github; print('✅ Core dependencies verified')"

# Final stage - runtime image
FROM python:3.11-slim as runtime

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy the application code - EXCLUDE problematic directories
COPY --chown=1000:1000 . .

# CRITICAL: Remove analysis cache and other problematic directories
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
RUN useradd --create-home --shell /bin/bash --uid 1000 app && \
    chown -R app:app /app
USER app

# Expose the port that the app runs on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Command to run the application
CMD ["python", "api_backend.py"]