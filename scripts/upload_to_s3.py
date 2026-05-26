"""
Upload IMDb dataset to S3 before running SageMaker notebook.
Usage: python scripts/upload_to_s3.py --bucket YOUR_BUCKET_NAME
"""
import boto3
import argparse
import os

def upload_dataset(bucket_name: str, local_path: str, s3_key: str = 'data/IMDB Dataset.csv'):
    s3 = boto3.client('s3')

    # Create bucket if it doesn't exist
    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' already exists")
    except Exception:
        print(f"Creating bucket '{bucket_name}'...")
        s3.create_bucket(Bucket=bucket_name)
        print("Bucket created")

    print(f"Uploading {local_path} → s3://{bucket_name}/{s3_key}")
    s3.upload_file(local_path, bucket_name, s3_key)
    print("Upload complete!")
    print(f"\nS3 path: s3://{bucket_name}/{s3_key}")
    print(f"\nUpdate S3_BUCKET in the SageMaker notebook to: {bucket_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upload IMDb dataset to S3')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--file', default='data/IMDB Dataset.csv', help='Local CSV path')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    local_path = os.path.join(project_dir, args.file)

    if not os.path.exists(local_path):
        print(f"File not found: {local_path}")
        exit(1)

    upload_dataset(args.bucket, local_path)
