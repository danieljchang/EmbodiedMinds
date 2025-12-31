#!/bin/bash
# S3 Sync Script for EB-Man Dataset
# Bucket: 11777-h1
# Region: us-east-2

set -e

BUCKET="11777-h1"
REGION="us-east-2"
LOCAL_DATA_DIR="$HOME/EmbodiedMinds/data/EB-Man_trajectory_dataset"
LOCAL_CHECKPOINTS_DIR="$HOME/EmbodiedMinds/checkpoints"
LOCAL_LOGS_DIR="$HOME/EmbodiedMinds/logs"

echo "=== S3 Sync Script ==="
echo "Bucket: s3://${BUCKET}"
echo "Region: ${REGION}"
echo ""

# Check AWS credentials
if ! aws sts get-caller-identity &>/dev/null; then
    echo "✗ AWS credentials not configured!"
    echo ""
    echo "Please either:"
    echo "  1. Attach an IAM role with S3 permissions to this EC2 instance, OR"
    echo "  2. Run: aws configure"
    echo ""
    exit 1
fi

echo "✓ AWS credentials verified"
echo ""

# Function to sync from S3 to local
sync_from_s3() {
    echo "=== Downloading from S3 ==="
    
    # Create local directories
    mkdir -p "${LOCAL_DATA_DIR}"
    mkdir -p "${LOCAL_CHECKPOINTS_DIR}"
    mkdir -p "${LOCAL_LOGS_DIR}"
    
    # Sync data directory
    if aws s3 ls "s3://${BUCKET}/data/" --region "${REGION}" &>/dev/null; then
        echo "Syncing data from s3://${BUCKET}/data/ to ${LOCAL_DATA_DIR}/"
        aws s3 sync "s3://${BUCKET}/data/" "${LOCAL_DATA_DIR}/" --region "${REGION}" --no-progress
        echo "✓ Data sync complete"
    else
        echo "⚠ No data/ directory found in bucket"
    fi
    
    # Sync checkpoints
    if aws s3 ls "s3://${BUCKET}/checkpoints/" --region "${REGION}" &>/dev/null; then
        echo "Syncing checkpoints from s3://${BUCKET}/checkpoints/ to ${LOCAL_CHECKPOINTS_DIR}/"
        aws s3 sync "s3://${BUCKET}/checkpoints/" "${LOCAL_CHECKPOINTS_DIR}/" --region "${REGION}" --no-progress
        echo "✓ Checkpoints sync complete"
    else
        echo "⚠ No checkpoints/ directory found in bucket"
    fi
    
    echo ""
}

# Function to sync to S3 from local
sync_to_s3() {
    echo "=== Uploading to S3 ==="
    
    # Sync checkpoints to S3
    if [ -d "${LOCAL_CHECKPOINTS_DIR}" ] && [ "$(ls -A ${LOCAL_CHECKPOINTS_DIR})" ]; then
        echo "Syncing checkpoints from ${LOCAL_CHECKPOINTS_DIR}/ to s3://${BUCKET}/checkpoints/"
        aws s3 sync "${LOCAL_CHECKPOINTS_DIR}/" "s3://${BUCKET}/checkpoints/" --region "${REGION}" --no-progress
        echo "✓ Checkpoints upload complete"
    else
        echo "⚠ No checkpoints to upload"
    fi
    
    # Sync logs to S3
    if [ -d "${LOCAL_LOGS_DIR}" ] && [ "$(ls -A ${LOCAL_LOGS_DIR})" ]; then
        echo "Syncing logs from ${LOCAL_LOGS_DIR}/ to s3://${BUCKET}/logs/"
        aws s3 sync "${LOCAL_LOGS_DIR}/" "s3://${BUCKET}/logs/" --region "${REGION}" --no-progress
        echo "✓ Logs upload complete"
    else
        echo "⚠ No logs to upload"
    fi
    
    echo ""
}

# Function to list S3 bucket contents
list_s3() {
    echo "=== S3 Bucket Contents ==="
    echo "Listing s3://${BUCKET}/"
    aws s3 ls "s3://${BUCKET}/" --region "${REGION}" --recursive --human-readable | head -50
    echo ""
}

# Main menu
case "${1}" in
    "download"|"from-s3"|"down")
        sync_from_s3
        ;;
    "upload"|"to-s3"|"up")
        sync_to_s3
        ;;
    "list"|"ls")
        list_s3
        ;;
    "both"|"sync")
        sync_from_s3
        sync_to_s3
        ;;
    *)
        echo "Usage: $0 {download|upload|list|both}"
        echo ""
        echo "Commands:"
        echo "  download, from-s3, down  - Download from S3 to EC2"
        echo "  upload, to-s3, up         - Upload from EC2 to S3"
        echo "  list, ls                  - List S3 bucket contents"
        echo "  both, sync                - Download and upload"
        echo ""
        exit 1
        ;;
esac

echo "=== Done ==="

