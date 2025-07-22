resource "aws_s3_bucket" "s3_bucket" {
  bucket = var.bucket_name
  acl = "private"
}

output "name" {
  value = aws_s3_bucket.s3_bucket.bucket

  #will need he arn later for premissions
  #  value = aws_s3_bucket.s3_bucket.arn

}
