import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# Read configuration from environment variables
S3_BUCKET = os.environ.get("S3_BUCKET", "fl-model-storage")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Create S3 client configured with Signature Version 4 for presigned URLs
s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    config=Config(signature_version="s3v4")
)

def get_bucket_name() -> str:
    """Returns the S3 bucket name."""
    return S3_BUCKET

def generate_presigned_download_url(object_key: str, expiration: int = 3600) -> str:
    """
    Generate a presigned URL to download a file from S3.
    """
    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": object_key},
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        print(f"Error generating presigned download URL: {e}")
        raise

def generate_presigned_upload_url(object_key: str, expiration: int = 3600) -> str:
    """
    Generate a presigned URL to upload a file to S3 via HTTP PUT.
    """
    try:
        url = s3_client.generate_presigned_url(
            "put_object",
            Params={"Bucket": S3_BUCKET, "Key": object_key},
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        print(f"Error generating presigned upload URL: {e}")
        raise

def upload_file(local_path: str, object_key: str) -> bool:
    """
    Upload a local file directly to S3.
    """
    try:
        s3_client.upload_file(local_path, S3_BUCKET, object_key)
        print(f"Successfully uploaded {local_path} to s3://{S3_BUCKET}/{object_key}")
        return True
    except ClientError as e:
        print(f"Error uploading file {local_path} to S3: {e}")
        return False

def download_file(object_key: str, local_path: str) -> bool:
    """
    Download a file from S3 to a local path.
    """
    try:
        # Ensure parent directories exist
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        s3_client.download_file(S3_BUCKET, object_key, local_path)
        print(f"Successfully downloaded s3://{S3_BUCKET}/{object_key} to {local_path}")
        return True
    except ClientError as e:
        print(f"Error downloading {object_key} from S3: {e}")
        return False
