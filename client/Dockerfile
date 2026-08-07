# Use the official CPU-optimized TensorFlow image
FROM tensorflow/tensorflow:2.15.0

# Set target directory
WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

# Install dependencies excluding tensorflow (which is already preinstalled in the base image)
RUN grep -v "tensorflow" requirements.txt > temp_req.txt && \
    pip install --no-cache-dir -r temp_req.txt && \
    rm temp_req.txt

# Copy source code and optional configuration files (if they exist)
COPY main.py model.py ./
COPY ip.txt* psswd.txt* ./

# Set command to execute the client script
ENTRYPOINT ["python", "-u", "main.py"]
