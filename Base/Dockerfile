# Dockerfile

# Lightweight Python image
FROM python:3.11-slim

# Avoid output buffering & pip cache
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Working directory inside the container
WORKDIR /app

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app
COPY app_ollama.py .

# Streamlit: don't try to open a browser in the container
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Internal port used by Streamlit
EXPOSE 8501

# Default command: run the Streamlit app
CMD ["streamlit", "run", "app_ollama.py", "--server.port=8501", "--server.address=0.0.0.0"]