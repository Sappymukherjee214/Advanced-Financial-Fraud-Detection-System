# Use standard slim python runtime
FROM python:3.11-slim

# Prevent python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed for compiling some python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy directories and files
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY app.py .
COPY main.py .

# Expose ports for Streamlit and FastAPI
EXPOSE 8501
EXPOSE 8000

# Default entrypoint starts Streamlit on container launch
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
