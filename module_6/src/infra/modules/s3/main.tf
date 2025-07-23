resource "aws_s3_bucket" "s3_bucket" {
  bucket = var.bucket_name
}

resource "aws_s3_bucket_public_access_block" "s3_bucket_pab" {
  bucket = aws_s3_bucket.s3_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "s3_bucket_acl_ownership" {
  bucket = aws_s3_bucket.s3_bucket.id

  rule {
    object_ownership = "BucketOwnerPreferred"
  }
}

resource "aws_s3_bucket_acl" "s3_bucket_acl" {
  depends_on = [
    aws_s3_bucket_ownership_controls.s3_bucket_acl_ownership,
    aws_s3_bucket_public_access_block.s3_bucket_pab,
  ]

  bucket = aws_s3_bucket.s3_bucket.id
  acl    = "private"
}

output "name" {
  value = aws_s3_bucket.s3_bucket.bucket

  #will need the arn later for permissions
  #  value = aws_s3_bucket.s3_bucket.arn
}
