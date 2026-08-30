import sys
import os
import json

# Append server directory to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))

import redis
from s3_helper import generate_presigned_download_url, generate_presigned_upload_url, get_bucket_name
from celery_app import aggregate_models_task, celery_app

def test_redis_connection():
    """
    Test connection to Redis and basic operations.
    """
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    client = redis.from_url(redis_url, protocol=2, decode_responses=True)
    
    # Ping Redis
    assert client.ping() is True
    print("Redis Ping Success.")
    
    # Test set/get
    client.set("test_key", "hello_redis")
    assert client.get("test_key") == "hello_redis"
    
    client.delete("test_key")
    print("Redis basic set/get and delete passed.")

def test_s3_presigned_urls():
    """
    Test generating presigned URLs without AWS credential issues.
    """
    bucket = get_bucket_name()
    assert bucket is not None
    assert len(bucket) > 0
    
    # Test GET URL generation
    download_url = generate_presigned_download_url("models/global/global_model_0.keras")
    assert download_url is not None
    assert "s3" in download_url.lower()
    assert "global_model_0.keras" in download_url
    print(f"Generated download presigned URL: {download_url[:60]}...")
    
    # Test PUT URL generation
    upload_url = generate_presigned_upload_url("models/round_1/client_test_model.keras")
    assert upload_url is not None
    assert "s3" in upload_url.lower()
    assert "client_test_model.keras" in upload_url
    print(f"Generated upload presigned URL: {upload_url[:60]}...")

def test_celery_task_registration():
    """
    Verify the Celery app is registered and the aggregation task exists.
    """
    assert celery_app is not None
    assert "tasks.aggregate_models" in celery_app.tasks
    print("Celery task registration verification passed.")

if __name__ == "__main__":
    print("Running S3 and Redis automated tests...")
    test_redis_connection()
    test_s3_presigned_urls()
    test_celery_task_registration()
    print("\nALL S3 AND REDIS AUTOMATED TESTS PASSED!")
