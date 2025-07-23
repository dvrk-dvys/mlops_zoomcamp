resource "aws_ecr_repository" "repo" {
  name                 = var.ecr_repo_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true  # Add this line

  image_scanning_configuration {
    scan_on_push = false
  }
}

# In practice, the image build-and-push step is handled separately by the CI/CD pipeline and not the IaC script.
# But because the lambda config would fail without an existing image URI in ECR,
# we can also upload any base image to bootstrap the lambda config, unrelated to your inference logic


resource null_resource ecr_image {
   triggers = {
     python_file = md5(file(var.lambda_function_local_path))
     docker_file = md5(file(var.docker_image_local_path))
   }

  provisioner "local-exec" {
    command = <<EOF
            aws ecr get-login-password --region ${var.region} | docker login --username AWS --password-stdin ${var.account_id}.dkr.ecr.${var.region}.amazonaws.com
            cd ../
            docker buildx rm lambda-builder || true
            docker buildx create --use --name lambda-builder || true
            docker buildx build --platform linux/amd64 --provenance=false --sbom=false --push -t ${aws_ecr_repository.repo.repository_url}:${var.ecr_image_tag} .
        EOF
  }
}

#In practice companies usually have two different repos for your application code and your base infrastructure
#The base infrastructure code generally contains base level components including your ecr repo which do not really need
# to change over time. However, the service side might contain some service specific IAC code such as your lambda config.
# Which may need to be updated when the docker image is already available


// Wait for the image to be uploaded, before lambda config runs
data aws_ecr_image lambda_image {
depends_on = [
  null_resource.ecr_image
]
repository_name = var.ecr_repo_name
image_tag       = var.ecr_image_tag
}

output "image_uri" {
 value     = "${aws_ecr_repository.repo.repository_url}:${data.aws_ecr_image.lambda_image.image_tag}"
}

# Temporary output for destroy - use a hardcoded tag
#output "image_uri" {
#  value     = "${aws_ecr_repository.repo.repository_url}:latest"
#}
