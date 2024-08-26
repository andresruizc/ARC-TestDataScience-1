FROM python:3.7-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Jupyter notebook and data file
COPY notebook/task1.ipynb .
COPY data/heart_failure_clinical_records_dataset.csv .
COPY src/main.py .

# Expose port for Jupyter
EXPOSE 8888

# Start Jupyter notebook
CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]