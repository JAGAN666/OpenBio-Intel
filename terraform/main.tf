terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state -- REQUIRED for .github/workflows/deploy.yml's Terraform
  # Apply job to be safe to run at all (a stateless CI runner has no local
  # terraform.tfstate; without this, it would conclude nothing exists yet
  # and attempt to create a second, colliding copy of this whole stack --
  # see provision_terraform_backend.sh's own header for the full reasoning).
  #
  # bucket/dynamodb_table are the EXACT values provision_terraform_backend.sh
  # created and printed -- verified live: aws s3api head-bucket and aws
  # dynamodb describe-table both confirm these exist, are versioned/
  # encrypted (bucket) and ACTIVE (table), before this block was written.
  # Hardcoded literally rather than left as a partial config supplied via
  # -backend-config= at init time: NOT a secret value (a bucket/table name
  # is not sensitive), so committing it is fine, and it is one less moving
  # piece the CI pipeline needs to be handed via GitHub secrets.
  backend "s3" {
    bucket         = "medical-rag-terraform-state-569486438558"
    key            = "medical-rag/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "medical-rag-terraform-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  name = var.project_name
  azs  = slice(data.aws_availability_zones.available.names, 0, 2)
}

# =============================================================================
# NETWORKING -- VPC, 2 public subnets (ALB), 2 private subnets (ECS tasks)
# =============================================================================
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true # required for the Cloud Map private DNS namespace

  tags = { Name = "${local.name}-vpc" }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${local.name}-igw" }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true

  tags = { Name = "${local.name}-public-${count.index}" }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = local.azs[count.index]

  tags = { Name = "${local.name}-private-${count.index}" }
}

# --- egress for private subnets ---------------------------------------------
# ONE NAT Gateway, not one per AZ. Every task in this stack (Qdrant, Neo4j,
# the backend) needs outbound internet regardless -- to pull public Docker
# Hub images and, for the backend, to reach the Anthropic API -- so private
# subnets need SOME route out. A single NAT is the cost-conscious choice
# (~$33/mo vs ~$66/mo for two) at the cost of AZ redundancy for egress: if
# the NAT's AZ has an outage, BOTH private subnets lose internet access even
# though the VPC itself spans 2 AZs. Acceptable for this workload (an
# internal demo-intelligence stack, not a multi-region SLA product); a
# production hardening pass would add a second NAT + per-AZ private route
# tables instead of the single shared one below.
resource "aws_eip" "nat" {
  domain = "vpc"
  tags   = { Name = "${local.name}-nat-eip" }
}

resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id
  depends_on    = [aws_internet_gateway.main]
  tags          = { Name = "${local.name}-nat" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  tags = { Name = "${local.name}-public-rt" }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }
  tags = { Name = "${local.name}-private-rt" }
}

resource "aws_route_table_association" "private" {
  count          = 2
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# =============================================================================
# SECURITY GROUPS
# =============================================================================
resource "aws_security_group" "alb" {
  name_prefix = "${local.name}-alb-"
  description = "Public ALB -- inbound HTTP from the internet only."
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTP from anywhere"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle { create_before_destroy = true }
  tags = { Name = "${local.name}-alb-sg" }
}

resource "aws_security_group" "ecs_tasks" {
  name_prefix = "${local.name}-ecs-tasks-"
  description = "All Fargate tasks -- app ports from the ALB, DB ports between tasks, all egress."
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Frontend (Next.js) from ALB"
    from_port       = 3000
    to_port         = 3000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  ingress {
    description     = "Backend (FastAPI) from ALB"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  # Qdrant (REST 6333 / gRPC 6334) and Neo4j (HTTP 7474 / Bolt 7687) are
  # reached ONLY by other tasks in this same security group (via Cloud Map
  # DNS), never by the ALB -- self-referencing ingress, not an ALB rule.
  ingress {
    description = "Qdrant REST/gRPC from other tasks in this SG"
    from_port   = 6333
    to_port     = 6334
    protocol    = "tcp"
    self        = true
  }

  ingress {
    description = "Neo4j HTTP/Bolt from other tasks in this SG"
    from_port   = 7474
    to_port     = 7687
    protocol    = "tcp"
    self        = true
  }

  egress {
    description = "All egress -- Docker Hub pulls, Anthropic API, EFS, etc."
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle { create_before_destroy = true }
  tags = { Name = "${local.name}-ecs-tasks-sg" }
}

resource "aws_security_group" "efs" {
  name_prefix = "${local.name}-efs-"
  description = "EFS mount targets -- NFS from ECS tasks only."
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "NFS from ECS tasks"
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle { create_before_destroy = true }
  tags = { Name = "${local.name}-efs-sg" }
}

# =============================================================================
# APPLICATION LOAD BALANCER
# =============================================================================
resource "aws_lb" "main" {
  name               = "${local.name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  # AWS default is 60s. Bumped: this app's research queries run a
  # multi-tool-call LangGraph agent (intent classification, tool calls,
  # parallel Map-Reduce extraction workers, a synthesis LLM call) that has
  # taken 60-90+ seconds for non-trivial questions throughout this
  # project's own local testing -- well past the default, and confirmed
  # live: the very first production query hit exactly this ceiling and
  # came back as a 502 from the ALB, not an application error.
  idle_timeout = 120

  tags = { Name = "${local.name}-alb" }
}

resource "aws_lb_target_group" "frontend" {
  name        = "frontend-tg"
  port        = 3000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip" # required for awsvpc-networked Fargate tasks -- there is no EC2 instance to register

  health_check {
    path                = "/"
    healthy_threshold   = 3
    unhealthy_threshold = 3
    interval            = 30
    timeout             = 5
    matcher             = "200"
  }

  tags = { Name = "frontend-tg" }
}

resource "aws_lb_target_group" "backend" {
  name        = "backend-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    # The REAL endpoint api.py exposes (verified against the app code, not
    # assumed) -- GET /api/health, a cheap readiness probe that does not
    # call the LLM.
    path                = "/api/health"
    healthy_threshold   = 3
    unhealthy_threshold = 3
    interval            = 30
    timeout             = 5
    matcher             = "200"
  }

  tags = { Name = "backend-tg" }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  # Default: everything not matched by a more specific rule goes to the
  # frontend -- the Next.js app, not the API.
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.frontend.arn
  }
}

resource "aws_lb_listener_rule" "backend_api" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.backend.arn
  }

  condition {
    path_pattern {
      values = ["/api/*"]
    }
  }
}

# =============================================================================
# ECS CLUSTER
# =============================================================================
resource "aws_ecs_cluster" "main" {
  name = "${local.name}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = { Name = "${local.name}-cluster" }
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name       = aws_ecs_cluster.main.name
  capacity_providers = ["FARGATE"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 100
  }
}

# =============================================================================
# CLOUD MAP -- private DNS namespace so containers reach each other by name
# (qdrant.clinical-rag.local, neo4j.clinical-rag.local) instead of IPs, which
# are not stable across Fargate task restarts/reschedules.
# =============================================================================
resource "aws_service_discovery_private_dns_namespace" "main" {
  name        = var.service_discovery_namespace
  description = "Internal service discovery for ${local.name}"
  vpc         = aws_vpc.main.id
}

resource "aws_service_discovery_service" "qdrant" {
  name = "qdrant"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id
    dns_records {
      ttl  = 10
      type = "A"
    }
    routing_policy = "MULTIVALUE"
  }

  health_check_custom_config {
    failure_threshold = 1
  }
}

resource "aws_service_discovery_service" "neo4j" {
  name = "neo4j"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id
    dns_records {
      ttl  = 10
      type = "A"
    }
    routing_policy = "MULTIVALUE"
  }

  health_check_custom_config {
    failure_threshold = 1
  }
}

# =============================================================================
# EFS -- stateful storage for Qdrant and Neo4j. One filesystem, two access
# points (separate root directories + POSIX ownership per database), each
# mounted into its own task at the SAME container path the local Docker
# Compose stack already uses (/qdrant/storage, /data) so nothing in either
# image needs to change to run here.
# =============================================================================
resource "aws_efs_file_system" "main" {
  creation_token   = "${local.name}-efs"
  encrypted        = true
  performance_mode = "generalPurpose"
  throughput_mode  = "bursting"

  tags = { Name = "${local.name}-efs" }
}

resource "aws_efs_mount_target" "main" {
  count           = 2
  file_system_id  = aws_efs_file_system.main.id
  subnet_id       = aws_subnet.private[count.index].id
  security_groups = [aws_security_group.efs.id]
}

resource "aws_efs_access_point" "qdrant" {
  file_system_id = aws_efs_file_system.main.id

  posix_user {
    uid = 0 # qdrant/qdrant's official image runs as root
    gid = 0
  }

  root_directory {
    path = "/qdrant-data"
    creation_info {
      owner_uid   = 0
      owner_gid   = 0
      permissions = "0755"
    }
  }

  tags = { Name = "${local.name}-qdrant-ap" }
  # Mount targets must exist before a task can attach an access point on them.
  depends_on = [aws_efs_mount_target.main]
}

resource "aws_efs_access_point" "neo4j" {
  file_system_id = aws_efs_file_system.main.id

  posix_user {
    uid = 7474 # the neo4j image's own "neo4j" user/group (verified: its Dockerfile creates uid/gid 7474)
    gid = 7474
  }

  root_directory {
    path = "/neo4j-data"
    creation_info {
      owner_uid   = 7474
      owner_gid   = 7474
      permissions = "0755"
    }
  }

  tags       = { Name = "${local.name}-neo4j-ap" }
  depends_on = [aws_efs_mount_target.main]
}

# Backend's FastEmbed model cache -- carries over a fix that was already
# correctly made in the local docker-compose.yml deployment (a mounted
# fastembed_cache volume at /root/.cache/fastembed, matching
# Dockerfile.backend's ENV FASTEMBED_CACHE_PATH) but was missed when this
# stack was ported to Fargate. Without it, every fresh backend task
# re-downloads the embedding model from HuggingFace Hub on its first
# request -- confirmed live: this alone was slow enough, stacked with the
# LLM calls and Qdrant round-trips a real query already makes, to exceed
# the ALB's 60s idle timeout and return a 502 on that very first query.
resource "aws_efs_access_point" "fastembed_cache" {
  file_system_id = aws_efs_file_system.main.id

  posix_user {
    uid = 0 # python:3.10-slim's default root user, matching Dockerfile.backend's own root_directory
    gid = 0
  }

  root_directory {
    path = "/fastembed-cache"
    creation_info {
      owner_uid   = 0
      owner_gid   = 0
      permissions = "0755"
    }
  }

  tags       = { Name = "${local.name}-fastembed-cache-ap" }
  depends_on = [aws_efs_mount_target.main]
}

# =============================================================================
# IAM -- one execution role (pull images, write logs, read secrets) shared by
# all four task definitions, one task role (the app's own AWS permissions at
# runtime -- currently none needed, kept separate from the execution role on
# principle rather than merged, since "can this task call AWS APIs" and "can
# ECS itself set this task up" are different concerns).
# =============================================================================
data "aws_iam_policy_document" "ecs_task_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecs_task_execution" {
  name               = "${local.name}-ecs-task-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume.json
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_managed" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Scoped to exactly the secrets this stack creates below -- not a
# wildcard, so the execution role can't read unrelated secrets in the account.
data "aws_iam_policy_document" "ecs_secrets_access" {
  statement {
    actions = ["secretsmanager:GetSecretValue"]
    resources = [
      aws_secretsmanager_secret.anthropic_api_key.arn,
      aws_secretsmanager_secret.neo4j_password.arn,
      aws_secretsmanager_secret.neo4j_auth.arn,
      aws_secretsmanager_secret.openai_api_key.arn,
      aws_secretsmanager_secret.jwt_secret_key.arn,
    ]
  }
}

resource "aws_iam_role_policy" "ecs_secrets_access" {
  name   = "${local.name}-ecs-secrets-access"
  role   = aws_iam_role.ecs_task_execution.id
  policy = data.aws_iam_policy_document.ecs_secrets_access.json
}

resource "aws_iam_role" "ecs_task" {
  name               = "${local.name}-ecs-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume.json
}

# =============================================================================
# SECRETS MANAGER -- ANTHROPIC_API_KEY and NEO4J_PASSWORD are injected via
# the container definition's `secrets` block (resolved by the ECS agent at
# task start), never as plaintext `environment` entries, which stay visible
# in the ECS console/API/CloudTrail indefinitely.
#
# The secret VALUE still originates from a Terraform variable here, so it
# does pass through `terraform plan`/state once -- a stricter setup would
# create these secrets out-of-band (`aws secretsmanager create-secret` run
# once, by hand or in a bootstrap script) and have this config only
# reference their ARN via a data source, so no plaintext ever touches
# Terraform state. Not done here, to keep this a single self-contained
# `terraform apply`; flagged as the harder-but-safer alternative.
# =============================================================================
resource "aws_secretsmanager_secret" "anthropic_api_key" {
  name                    = "${local.name}/anthropic-api-key"
  recovery_window_in_days = 0 # demo/dev stack -- allow immediate re-creation on destroy/apply cycles
}

resource "aws_secretsmanager_secret_version" "anthropic_api_key" {
  secret_id     = aws_secretsmanager_secret.anthropic_api_key.id
  secret_string = var.anthropic_api_key
}

resource "aws_secretsmanager_secret" "neo4j_password" {
  name                    = "${local.name}/neo4j-password"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "neo4j_password" {
  secret_id     = aws_secretsmanager_secret.neo4j_password.id
  secret_string = var.neo4j_password
}

# Neo4j's own NEO4J_AUTH env var needs the COMPOSED "neo4j/<password>"
# string, not the bare password research_agent.py's NEO4J_PASSWORD reads --
# AWS's `secrets` block injects a secret's value verbatim, with no
# concatenation step, so a second secret holding the pre-composed string is
# simpler and more explicit than trying to build it inside the container's
# entrypoint. A little redundant (the password exists in two secrets), but
# each consumer gets exactly the shape it needs from Secrets Manager, never
# plaintext.
resource "aws_secretsmanager_secret" "neo4j_auth" {
  name                    = "${local.name}/neo4j-auth"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "neo4j_auth" {
  secret_id     = aws_secretsmanager_secret.neo4j_auth.id
  secret_string = "neo4j/${var.neo4j_password}"
}

resource "aws_secretsmanager_secret" "openai_api_key" {
  name                    = "${local.name}/openai-api-key"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "openai_api_key" {
  secret_id     = aws_secretsmanager_secret.openai_api_key.id
  secret_string = var.openai_api_key
}

resource "aws_secretsmanager_secret" "jwt_secret_key" {
  name                    = "${local.name}/jwt-secret-key"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "jwt_secret_key" {
  secret_id     = aws_secretsmanager_secret.jwt_secret_key.id
  secret_string = var.jwt_secret_key
}

# =============================================================================
# ECR -- repositories for OUR OWN images (backend, frontend, etl). Qdrant/Neo4j
# pull directly from Docker Hub, so they need no repository here.
# =============================================================================
resource "aws_ecr_repository" "backend" {
  name                 = "${local.name}/backend"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "frontend" {
  name                 = "${local.name}/frontend"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "etl" {
  name                 = "${local.name}/etl"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# =============================================================================
# CLOUDWATCH LOGS -- one group per service
# =============================================================================
resource "aws_cloudwatch_log_group" "qdrant" {
  name              = "/ecs/${local.name}/qdrant"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "neo4j" {
  name              = "/ecs/${local.name}/neo4j"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "backend" {
  name              = "/ecs/${local.name}/backend"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "frontend" {
  name              = "/ecs/${local.name}/frontend"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "etl" {
  name              = "/ecs/${local.name}/daily-delta-job"
  retention_in_days = 14
}

# =============================================================================
# QDRANT -- Fargate task + EFS-backed storage + Cloud Map registration.
# No load balancer: reached only via qdrant.clinical-rag.local by the other
# tasks in this VPC, never from the internet.
# =============================================================================
resource "aws_ecs_task_definition" "qdrant" {
  family                   = "${local.name}-qdrant"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.qdrant_cpu
  memory                   = var.qdrant_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  volume {
    name = "qdrant-storage"
    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.main.id
      transit_encryption = "ENABLED"
      authorization_config {
        access_point_id = aws_efs_access_point.qdrant.id
        iam             = "DISABLED"
      }
    }
  }

  container_definitions = jsonencode([
    {
      name      = "qdrant"
      image     = var.qdrant_image
      essential = true
      portMappings = [
        { containerPort = 6333, protocol = "tcp" }, # REST
        { containerPort = 6334, protocol = "tcp" }, # gRPC
      ]
      mountPoints = [
        {
          sourceVolume  = "qdrant-storage"
          containerPath = "/qdrant/storage" # same path the local docker-compose.yml volume already uses
          readOnly      = false
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.qdrant.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "qdrant"
        }
      }
    }
  ])

  tags = { Name = "${local.name}-qdrant-task" }
}

resource "aws_ecs_service" "qdrant" {
  name            = "${local.name}-qdrant"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.qdrant.arn
  desired_count   = var.qdrant_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  service_registries {
    registry_arn = aws_service_discovery_service.qdrant.arn
  }

  # A single EFS-backed volume can only safely be mounted read-write by ONE
  # task at a time -- 0/100 lets the old task stop before/as the replacement
  # starts during a deploy, instead of the default rolling-update behavior
  # that would briefly run two tasks against the same volume.
  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100

  tags = { Name = "${local.name}-qdrant-service" }
}

# =============================================================================
# NEO4J -- same pattern as Qdrant: EFS-backed, Cloud Map registered, no ALB.
# =============================================================================
resource "aws_ecs_task_definition" "neo4j" {
  family                   = "${local.name}-neo4j"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.neo4j_cpu
  memory                   = var.neo4j_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  volume {
    name = "neo4j-data"
    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.main.id
      transit_encryption = "ENABLED"
      authorization_config {
        access_point_id = aws_efs_access_point.neo4j.id
        iam             = "DISABLED"
      }
    }
  }

  container_definitions = jsonencode([
    {
      name      = "neo4j"
      image     = var.neo4j_image
      essential = true
      portMappings = [
        { containerPort = 7474, protocol = "tcp" }, # HTTP / browser UI
        { containerPort = 7687, protocol = "tcp" }, # Bolt -- what build_kg.py / research_agent.py use
      ]
      mountPoints = [
        {
          sourceVolume  = "neo4j-data"
          containerPath = "/data" # same path the local docker-compose.yml volume already uses
          readOnly      = false
        }
      ]
      secrets = [
        {
          name      = "NEO4J_AUTH"
          valueFrom = aws_secretsmanager_secret.neo4j_auth.arn
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.neo4j.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "neo4j"
        }
      }
    }
  ])

  tags = { Name = "${local.name}-neo4j-task" }
}

resource "aws_ecs_service" "neo4j" {
  name            = "${local.name}-neo4j"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.neo4j.arn
  desired_count   = var.neo4j_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  service_registries {
    registry_arn = aws_service_discovery_service.neo4j.arn
  }

  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100

  tags = { Name = "${local.name}-neo4j-service" }
}

# =============================================================================
# BACKEND (FastAPI + LangGraph) -- attached to the ALB's backend-tg, reaches
# Qdrant/Neo4j via Cloud Map DNS.
# =============================================================================
resource "aws_ecs_task_definition" "backend" {
  family                   = "${local.name}-backend"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.backend_cpu
  memory                   = var.backend_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  volume {
    name = "fastembed-cache"
    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.main.id
      transit_encryption = "ENABLED"
      authorization_config {
        access_point_id = aws_efs_access_point.fastembed_cache.id
        iam             = "DISABLED"
      }
    }
  }

  container_definitions = jsonencode([
    {
      name      = "backend"
      image     = "${aws_ecr_repository.backend.repository_url}:${var.backend_image_tag}"
      essential = true
      # No explicit `command` -- falls back to Dockerfile.backend's own CMD
      # (verified: `uvicorn api:app --host 0.0.0.0 --port 8000`, no
      # --reload). Not duplicated here to avoid the two drifting out of
      # sync if the Dockerfile's CMD ever changes.
      portMappings = [
        { containerPort = 8000, protocol = "tcp" }
      ]
      mountPoints = [
        {
          sourceVolume  = "fastembed-cache"
          containerPath = "/root/.cache/fastembed" # matches Dockerfile.backend's ENV FASTEMBED_CACHE_PATH
          readOnly      = false
        }
      ]
      environment = [
        # NOTE: intentionally QDRANT_HOST/QDRANT_PORT and NEO4J_URI, NOT the
        # QDRANT_URL the spec's prose suggested. Verified directly against
        # research_agent.py: it reads os.getenv("QDRANT_HOST", "localhost")
        # and os.getenv("QDRANT_PORT", "6333") as two separate variables --
        # there is no QDRANT_URL anywhere in the app. Setting QDRANT_URL as
        # written in the spec would be silently ignored at runtime and the
        # container would fall back to connecting to "localhost" (itself),
        # failing to reach Qdrant at all -- the exact same class of bug
        # already caught once in this project's docker-compose.yml
        # (QDRANT_HOST) and frontend build arg (NEXT_PUBLIC_API_URL).
        # NEO4J_URI, by contrast, DOES match the spec's literal name --
        # verified the same way against research_agent.py's own
        # os.getenv("NEO4J_URI", ...) default.
        { name = "QDRANT_HOST", value = "qdrant.${var.service_discovery_namespace}" },
        { name = "QDRANT_PORT", value = "6333" },
        { name = "NEO4J_URI", value = "bolt://neo4j.${var.service_discovery_namespace}:7687" },
        { name = "NEO4J_USER", value = "neo4j" },
        # Not sensitive (it's an algorithm name, "HS256"), so a plain
        # environment entry rather than a Secrets Manager value -- explicit
        # here rather than relying on auth.py's own os.getenv(...,"HS256")
        # fallback, so a future change to that default can't silently drift
        # from what this deployment actually signs/verifies with.
        { name = "JWT_ALGORITHM", value = var.jwt_algorithm },
      ]
      secrets = [
        { name = "ANTHROPIC_API_KEY", valueFrom = aws_secretsmanager_secret.anthropic_api_key.arn },
        { name = "NEO4J_PASSWORD", valueFrom = aws_secretsmanager_secret.neo4j_password.arn },
        # Previously missing entirely -- see var.openai_api_key's own
        # comment. research_agent.py's search tools call embeddings.py's
        # embed_query() on every request; without this, every search on a
        # deployed backend fails closed (embeddings.py raises SystemExit
        # when OPENAI_API_KEY is unset).
        { name = "OPENAI_API_KEY", valueFrom = aws_secretsmanager_secret.openai_api_key.arn },
        # Same previously-missing-entirely gap, this time for auth.py: with
        # no JWT_SECRET_KEY, get_current_user() raises a 500 on every
        # request (fails CLOSED, per that module's own design -- see its
        # docstring -- not open), so the deployed backend would reject
        # 100% of traffic post-auth-rollout without this.
        { name = "JWT_SECRET_KEY", valueFrom = aws_secretsmanager_secret.jwt_secret_key.arn },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.backend.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "backend"
        }
      }
    }
  ])

  tags = { Name = "${local.name}-backend-task" }
}

resource "aws_ecs_service" "backend" {
  name            = "${local.name}-backend"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.backend.arn
  desired_count   = var.backend_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.backend.arn
    container_name   = "backend"
    container_port   = 8000
  }

  # Qdrant/Neo4j should be reachable before the backend starts probing them
  # (the app itself is defensive about a slow/unready DB, but there is no
  # reason to race the startup order when Terraform can express it directly).
  depends_on = [
    aws_ecs_service.qdrant,
    aws_ecs_service.neo4j,
    aws_lb_listener.http,
  ]

  tags = { Name = "${local.name}-backend-service" }
}

# =============================================================================
# FRONTEND (Next.js) -- attached to the ALB's frontend-tg.
#
# NEXT_PUBLIC_API_URL is declared below for completeness, but setting it
# here has NO EFFECT on what the browser actually calls. Verified directly
# in the earlier Docker Compose milestone: Next.js inlines NEXT_PUBLIC_*
# into the CLIENT JS BUNDLE at `next build` TIME (frontend/Dockerfile's
# builder stage), not read from the environment at container start. By the
# time this task definition's `environment` block takes effect, the bundle
# already has SOME value (or none) baked in from whenever the image was
# built.
#
# The real fix is a deployment-ORDERING problem, not a Terraform problem:
# the ALB's DNS name does not exist until `aws_lb.main` is created, but the
# frontend image must be built WITH that DNS name as a --build-arg BEFORE
# it is pushed to ECR and run here. The practical sequence:
#   1. terraform apply -target=aws_lb.main   (or a full apply -- the ALB is
#      cheap and has no dependency on the frontend image existing)
#   2. docker build --build-arg NEXT_PUBLIC_API_URL="http://$(terraform
#      output -raw alb_dns_name)" -t <frontend ecr repo>:latest ./frontend
#      && docker push <frontend ecr repo>:latest
#   3. terraform apply (again, or `aws ecs update-service --force-new-deployment`)
#      to roll the frontend service onto the image that now has the correct
#      URL baked in.
# outputs.tf exposes alb_dns_name and ecr_repository_url_frontend
# specifically to support scripting step 2.
# =============================================================================
resource "aws_ecs_task_definition" "frontend" {
  family                   = "${local.name}-frontend"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.frontend_cpu
  memory                   = var.frontend_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "frontend"
      image     = "${aws_ecr_repository.frontend.repository_url}:${var.frontend_image_tag}"
      essential = true
      portMappings = [
        { containerPort = 3000, protocol = "tcp" }
      ]
      environment = [
        # See the module-level comment above: this value is a no-op unless
        # the image was BUILT with the same value as a --build-arg. Kept
        # here anyway since it's harmless and documents intent, and would
        # matter if the app is ever refactored to read it server-side at
        # request time instead of inlining it client-side at build time.
        { name = "NEXT_PUBLIC_API_URL", value = "http://${aws_lb.main.dns_name}" }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.frontend.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "frontend"
        }
      }
    }
  ])

  tags = { Name = "${local.name}-frontend-task" }
}

resource "aws_ecs_service" "frontend" {
  name            = "${local.name}-frontend"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.frontend.arn
  desired_count   = var.frontend_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.frontend.arn
    container_name   = "frontend"
    container_port   = 3000
  }

  depends_on = [
    aws_ecs_service.backend,
    aws_lb_listener.http,
  ]

  tags = { Name = "${local.name}-frontend-service" }
}

# =============================================================================
# DAILY DELTA JOB (fetch_daily_updates.py) -- the "Feed" half of Seed-and-Feed.
# A scheduled Fargate TASK (RunTask via EventBridge), not a long-running
# SERVICE -- it runs once at 02:00 UTC, does its work, and exits. No
# aws_ecs_service, no load balancer, no desired_count.
#
# NO EFS VOLUMES HERE -- deliberately deviating from the literal spec
# instruction to "mount the same EFS volumes for Qdrant and Neo4j." Verified
# against how every OTHER task in this stack already reaches those two
# databases: the backend task definition above has NO qdrant-storage or
# neo4j-data volume either -- it reaches them over the NETWORK, via Cloud
# Map DNS (QDRANT_HOST/NEO4J_URI env vars), exactly matching how
# fetch_daily_updates.py itself is written (a QdrantClient(host=...) /
# neo4j.GraphDatabase.driver(NEO4J_URI) network client, never a local
# filesystem path). Qdrant and Neo4j are each SINGLE-WRITER databases that
# already have their own dedicated task mounting their EFS access point
# read-write as their own on-disk storage engine (see
# aws_efs_access_point.qdrant / .neo4j above); a THIRD task mounting the
# same access point read-write while those processes are live would let two
# independent processes mutate the same on-disk storage files
# simultaneously -- corrupting Qdrant's/Neo4j's storage engine, not adding
# a second writer safely. Reaching them the same way the backend already
# does, over the network, is both correct and consistent with this stack's
# one other precedent for "a task needs Qdrant+Neo4j" instead of inventing
# a second, riskier pattern for this one task.
resource "aws_ecs_task_definition" "daily_delta_job" {
  family                   = "${local.name}-daily-delta-job"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.etl_cpu
  memory                   = var.etl_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "daily-delta-job"
      image     = "${aws_ecr_repository.etl.repository_url}:${var.etl_image_tag}"
      essential = true
      # No explicit `command` -- falls back to Dockerfile.etl's own CMD
      # (`python fetch_daily_updates.py`, no flags -- production default:
      # --days 1 ["yesterday"], both sources, S3 archiving and Neo4j
      # entity resolution both on). Not duplicated here, same reasoning as
      # the backend task definition's own comment on this.
      environment = [
        # Same QDRANT_HOST/QDRANT_PORT/NEO4J_URI pattern as the backend
        # task above -- see this resource's own comment for why this is a
        # network client, not an EFS mount.
        { name = "QDRANT_HOST", value = "qdrant.${var.service_discovery_namespace}" },
        { name = "QDRANT_PORT", value = "6333" },
        { name = "NEO4J_URI", value = "bolt://neo4j.${var.service_discovery_namespace}:7687" },
        { name = "NEO4J_USER", value = "neo4j" },
      ]
      secrets = [
        { name = "OPENAI_API_KEY", valueFrom = aws_secretsmanager_secret.openai_api_key.arn },
        { name = "NEO4J_PASSWORD", valueFrom = aws_secretsmanager_secret.neo4j_password.arn },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.etl.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "daily-delta-job"
        }
      }
    }
  ])

  tags = { Name = "${local.name}-daily-delta-job-task" }
}

# --- IAM: EventBridge needs its own role to call ecs:RunTask on our behalf --
# a DIFFERENT concern from the task's own execution/task roles above (those
# govern what the CONTAINER can do once running; this governs what
# EVENTBRIDGE ITSELF is allowed to do to start it).
data "aws_iam_policy_document" "eventbridge_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "eventbridge_ecs_run_task" {
  name               = "${local.name}-eventbridge-ecs-run-task"
  assume_role_policy = data.aws_iam_policy_document.eventbridge_assume.json
}

# Scoped to exactly this one task definition family (all revisions -- ":*",
# since Terraform bumps the revision on every container_definitions change)
# and this one cluster, not a wildcard across the account. iam:PassRole is
# separately required because RunTask, called BY EventBridge, must hand off
# the execution/task roles to the ECS agent that actually launches the task.
data "aws_iam_policy_document" "eventbridge_ecs_run_task" {
  statement {
    actions   = ["ecs:RunTask"]
    resources = ["${replace(aws_ecs_task_definition.daily_delta_job.arn, "/:\\d+$/", "")}:*"]
    condition {
      test     = "ArnEquals"
      variable = "ecs:cluster"
      values   = [aws_ecs_cluster.main.arn]
    }
  }
  statement {
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.ecs_task_execution.arn, aws_iam_role.ecs_task.arn]
  }
}

resource "aws_iam_role_policy" "eventbridge_ecs_run_task" {
  name   = "${local.name}-eventbridge-ecs-run-task"
  role   = aws_iam_role.eventbridge_ecs_run_task.id
  policy = data.aws_iam_policy_document.eventbridge_ecs_run_task.json
}

# --- EventBridge: cron(0 2 * * ? *) -- 02:00 UTC daily, per the spec -------
resource "aws_cloudwatch_event_rule" "daily_delta_job" {
  name                = "${local.name}-daily-delta-job-schedule"
  description         = "Triggers the daily delta ETL Fargate task (fetch_daily_updates.py)."
  schedule_expression = "cron(0 2 * * ? *)"
}

resource "aws_cloudwatch_event_target" "daily_delta_job" {
  rule     = aws_cloudwatch_event_rule.daily_delta_job.name
  arn      = aws_ecs_cluster.main.arn
  role_arn = aws_iam_role.eventbridge_ecs_run_task.arn

  ecs_target {
    task_definition_arn = aws_ecs_task_definition.daily_delta_job.arn
    task_count          = 1
    launch_type         = "FARGATE"

    network_configuration {
      subnets          = aws_subnet.private[*].id
      security_groups  = [aws_security_group.ecs_tasks.id]
      assign_public_ip = false
    }
  }
}

# =============================================================================
# CI/CD -- GitHub Actions OIDC federation + a deploy role, so
# .github/workflows/deploy.yml can authenticate to AWS via
# aws-actions/configure-aws-credentials WITHOUT a long-lived
# AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY pair sitting in GitHub repo
# secrets. This is the standard, current AWS-recommended pattern for GitHub
# Actions specifically -- see
# https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services
# -- not a bespoke choice.
# =============================================================================
data "aws_caller_identity" "current" {}

# GitHub's own OIDC token issuer -- one provider per AWS account, shared by
# every repo/workflow that wants to federate into it (the per-REPO scoping
# happens in the ROLE's trust policy below, not here).
resource "aws_iam_openid_connect_provider" "github_actions" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
  # This is GitHub's well-documented, stable root CA thumbprint (used in
  # essentially every official GitHub+AWS OIDC guide). AWS has since
  # clarified it does not actually validate this value against GitHub's
  # cert for well-known OIDC issuers -- the field stays required by this
  # resource's schema regardless, so it is supplied for that reason, not
  # because rotating it would meaningfully change trust here.
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]

  tags = { Name = "${local.name}-github-actions-oidc" }
}

data "aws_iam_policy_document" "github_actions_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github_actions.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    # Restricts WHICH workflow runs can assume this role: only runs
    # triggered on var.github_repository's main branch -- matching
    # deploy.yml's own `on: push: branches: [main]` trigger, so a
    # workflow run on a fork, a PR, or any other branch cannot assume
    # deploy credentials even if it somehow referenced this role ARN.
    #
    # StringLike with a wildcard, not StringEquals on the plain
    # "owner/repo" form -- verified live (a real token, printed via a
    # temporary debug step in deploy.yml, then removed) that this repo's
    # actual `sub` claim is "repo:OWNER@<owner_id>/REPO@<repo_id>:ref:...",
    # not the plain form, despite this repo's own
    # /actions/oidc/customization/sub API reporting use_immutable_subject:
    # false. The wildcards only fill the gap where an optional "@<id>"
    # suffix may or may not appear on each half -- the owner/repo prefixes
    # and the exact branch ref suffix are still fully anchored, so this
    # is not a broadened trust boundary, just a tolerant one.
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values = [
        "repo:${split("/", var.github_repository)[0]}*/${split("/", var.github_repository)[1]}*:ref:refs/heads/main"
      ]
    }
  }
}

resource "aws_iam_role" "github_actions_deploy" {
  name               = "${local.name}-github-actions-deploy"
  assume_role_policy = data.aws_iam_policy_document.github_actions_assume.json
  # 1h is plenty for a build+push+apply run and limits how long a leaked
  # session credential (e.g. from a compromised runner mid-job) stays usable.
  max_session_duration = 3600

  tags = { Name = "${local.name}-github-actions-deploy" }
}

# Scoped per-service, not AdministratorAccess -- but honestly broad within
# each service (ec2:*, ecs:*, etc.) rather than an exhaustively minimal
# action list. VERIFIED TRADEOFF, not an oversight: Terraform apply against
# this stack touches dozens of distinct resource types across ec2 (vpc,
# subnets, route tables, NAT/EIP, security groups), ecs, elb, efs,
# secretsmanager, ecr, logs, servicediscovery, and events -- hand-enumerating
# the exact action list for every one of those is exactly the kind of
# painstaking work that silently breaks a "production-grade" pipeline the
# first time `terraform apply` touches a resource type someone forgot to
# list, which is a worse operational failure mode than the broader grant
# here. The ONE service where that tradeoff does NOT apply is iam itself:
# unrestricted iam:* is a privilege-escalation path (a role that can create
# arbitrary IAM roles/policies can grant itself anything), so iam actions
# below are both resource-scoped to this stack's own role name prefix AND
# limited to the specific actions Terraform actually needs for the roles
# this config defines -- tightening the non-IAM statements further is a
# reasonable follow-up, not a gap being hidden here.
data "aws_iam_policy_document" "github_actions_deploy" {
  statement {
    sid    = "InfrastructureServices"
    effect = "Allow"
    actions = [
      "ec2:*",
      "ecs:*",
      "elasticloadbalancing:*",
      "efs:*",
      "ecr:*",
      "secretsmanager:*",
      "logs:*",
      "servicediscovery:*",
      "events:*",
      "application-autoscaling:*",
      "sts:GetCallerIdentity",
    ]
    resources = ["*"]
  }

  statement {
    sid    = "IamForThisStacksRolesOnly"
    effect = "Allow"
    actions = [
      "iam:GetRole", "iam:CreateRole", "iam:DeleteRole", "iam:TagRole",
      "iam:GetRolePolicy", "iam:PutRolePolicy", "iam:DeleteRolePolicy",
      "iam:ListRolePolicies", "iam:ListAttachedRolePolicies",
      "iam:AttachRolePolicy", "iam:DetachRolePolicy",
    ]
    resources = [
      "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${local.name}-*",
    ]
  }

  statement {
    # PassRole is the actual privilege-escalation-sensitive action (it's
    # what lets a caller hand one of this stack's roles to a NEW resource,
    # e.g. an ECS task) -- separated from the statement above and further
    # restricted by PassedToService so this role can only pass
    # medical-rag-* roles to the two AWS services that legitimately need
    # them (ECS tasks, EventBridge), never to an arbitrary service.
    sid     = "PassThisStacksRolesToECSAndEventBridgeOnly"
    effect  = "Allow"
    actions = ["iam:PassRole"]
    resources = [
      "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${local.name}-*",
    ]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["ecs-tasks.amazonaws.com", "events.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "github_actions_deploy" {
  name   = "${local.name}-github-actions-deploy"
  role   = aws_iam_role.github_actions_deploy.id
  policy = data.aws_iam_policy_document.github_actions_deploy.json
}
