FROM python:3.10-slim

WORKDIR /workspace

# Install system utilities
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements entirely
COPY requirements.txt .

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the project files
COPY . .

# Expose port (Hugging Face standard)
EXPOSE 7860

# Run gunicorn inside the proper directory
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "--workers", "2", "app.app:app"]
