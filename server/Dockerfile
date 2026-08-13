FROM tensorflow/tensorflow:2.15.0

# Install official Redis 7+ repository and packages
RUN apt-get update && apt-get install -y lsb-release curl gpg && \
    curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list && \
    apt-get update && apt-get install -y redis-server redis-tools

WORKDIR /app

COPY requirements.txt .

RUN grep -v "tensorflow" requirements.txt > temp_req.txt && \
    pip install --no-cache-dir -r temp_req.txt && \
    rm temp_req.txt

COPY . .

EXPOSE 8000 6379

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
