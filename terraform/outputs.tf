output "alb_dns_name" {
  description = <<-EOT
    Public DNS name of the ALB -- the app's entry point (http://<this>).
    Also REQUIRED as the --build-arg NEXT_PUBLIC_API_URL value when building
    the frontend image (see the comment on aws_ecs_task_definition.frontend
    in main.tf for why this can't just be a runtime environment variable).
  EOT
  value       = aws_lb.main.dns_name
}

output "ecr_repository_url_backend" {
  description = "Push the backend image here: docker push <this>:latest"
  value       = aws_ecr_repository.backend.repository_url
}

output "ecr_repository_url_frontend" {
  description = "Push the frontend image here: docker push <this>:latest"
  value       = aws_ecr_repository.frontend.repository_url
}

output "ecr_repository_url_etl" {
  description = "Push the daily-delta-job ETL image here: docker push <this>:latest"
  value       = aws_ecr_repository.etl.repository_url
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "service_discovery_namespace" {
  description = "Cloud Map namespace -- internal service names resolve as <service>.<this>."
  value       = var.service_discovery_namespace
}

output "vpc_id" {
  value = aws_vpc.main.id
}

output "private_subnet_ids" {
  value = aws_subnet.private[*].id
}

output "public_subnet_ids" {
  value = aws_subnet.public[*].id
}

output "efs_file_system_id" {
  value = aws_efs_file_system.main.id
}

output "github_actions_deploy_role_arn" {
  description = <<-EOT
    Set this as the AWS_DEPLOY_ROLE_ARN secret in the GitHub repo running
    .github/workflows/deploy.yml (Settings -> Secrets and variables ->
    Actions). Only assumable by workflow runs from var.github_repository's
    main branch -- see aws_iam_role.github_actions_deploy's trust policy.
  EOT
  value       = aws_iam_role.github_actions_deploy.arn
}
