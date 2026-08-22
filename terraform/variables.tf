variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Prefix applied to every resource name/tag."
  type        = string
  default     = "medical-rag"
}

# --- CI/CD (GitHub Actions OIDC -> IAM role, see main.tf's github_actions_deploy) --
variable "github_repository" {
  # Scopes the OIDC trust policy's `sub` condition so ONLY workflow runs
  # from this exact repo (on main) can assume the deploy role -- an empty
  # default is intentional: applying with this unset would produce a trust
  # policy matching NO real repo (a syntactically valid but permanently
  # unassumable role), which fails safely, rather than a wildcard that
  # would trust every GitHub repo on the internet.
  description = "\"<github-org-or-user>/<repo-name>\" -- must match the repo hosting .github/workflows/deploy.yml."
  type        = string
  default     = ""
}

# --- networking --------------------------------------------------------------
variable "vpc_cidr" {
  description = "CIDR block for the VPC."
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDRs for the 2 public subnets (ALB)."
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDRs for the 2 private subnets (ECS tasks)."
  type        = list(string)
  default     = ["10.0.11.0/24", "10.0.12.0/24"]
}

# --- service discovery ---------------------------------------------------------
variable "service_discovery_namespace" {
  description = "Cloud Map private DNS namespace -- containers reach each other at <service>.<this>."
  type        = string
  default     = "clinical-rag.local"
}

# --- images --------------------------------------------------------------------
# Qdrant and Neo4j are public images pulled straight from Docker Hub -- no ECR
# repo needed for them. Backend/frontend are OUR images; this config creates
# their ECR repositories, but does NOT build or push into them -- that is a
# separate `docker build && docker push` step (see outputs.tf), because
# Terraform provisions infrastructure, it does not build application images.
variable "qdrant_image" {
  type    = string
  default = "qdrant/qdrant:latest"
}

variable "neo4j_image" {
  type    = string
  default = "neo4j:5"
}

variable "backend_image_tag" {
  description = "Tag to deploy from the backend ECR repo (pushed out-of-band)."
  type        = string
  default     = "latest"
}

variable "frontend_image_tag" {
  description = "Tag to deploy from the frontend ECR repo (pushed out-of-band)."
  type        = string
  default     = "latest"
}

# --- Fargate sizing --------------------------------------------------------
# Valid Fargate (cpu, memory) combinations only -- 1024 cpu (1 vCPU) allows
# 2048-8192 MB in 1024 MB steps, 512 cpu (.5 vCPU) allows 1024-4096 MB in
# 1024 MB steps, 256 cpu (.25 vCPU) allows 512-2048 MB in 1024 MB steps.
# Qdrant/Neo4j sized generously (both are memory-hungry -- Neo4j especially
# wants JVM heap); frontend is the lightest (a static-ish Next.js server).
variable "qdrant_cpu" {
  type    = number
  default = 1024
}

variable "qdrant_memory" {
  type    = number
  default = 2048
}

variable "neo4j_cpu" {
  type    = number
  default = 1024
}

variable "neo4j_memory" {
  type    = number
  default = 3072
}

variable "backend_cpu" {
  type    = number
  default = 1024
}

variable "backend_memory" {
  # Bumped from the original 512/1024 -- confirmed via a live OOM kill
  # ("OutOfMemoryError: container killed due to memory usage", exit 137) on
  # the very first real research query: loading the FastEmbed/onnxruntime
  # embedding model into memory, stacked on FastAPI/LangGraph/uvicorn's own
  # footprint, exceeded 1024MB even for a single query (not a batch). 3072
  # matches what a one-off ETL task needed headroom for at a much heavier
  # 50-document batch load (which itself OOM'd at 1024 and 3072 before
  # succeeding at 8192) -- generous rather than tight, since a second OOM
  # kill here means another silent multi-minute outage.
  type    = number
  default = 3072
}

variable "frontend_cpu" {
  type    = number
  default = 256
}

variable "frontend_memory" {
  type    = number
  default = 512
}

variable "etl_cpu" {
  type    = number
  default = 1024
}

variable "etl_memory" {
  # 4096, not the 2048 a "small daily delta" might suggest at a glance.
  # This project has direct, documented history with an ETL-shaped batch
  # workload OOM'ing at 1024 AND 3072 before succeeding at 8192 (see
  # backend_memory's own comment above) -- that was a heavier one-off load
  # than a routine day's delta, but the RxNorm entity-resolution step this
  # task runs (scispacy + spacy + nmslib + scikit-learn, a ~107,000-concept
  # knowledge base loaded into memory every invocation) has real baseline
  # memory cost regardless of how few records a given day's delta contains.
  # 4096 is a middle ground between that documented failure history and
  # this job's genuinely lighter routine payload -- tune up toward 8192 if
  # a real run OOMs, rather than assuming this default is proven sufficient.
  type    = number
  default = 4096
}

variable "etl_image_tag" {
  description = "Tag to deploy from the ETL ECR repo (pushed out-of-band)."
  type        = string
  default     = "latest"
}

# --- secrets -----------------------------------------------------------------
# Passed as sensitive Terraform variables (supply via a gitignored
# terraform.tfvars or TF_VAR_* env vars -- never commit real values) and
# stored in AWS Secrets Manager, injected into the task via the container
# definition's `secrets` block rather than plaintext `environment`. Plaintext
# task-definition env vars are visible in the ECS console and CloudTrail
# indefinitely; Secrets Manager values are not. A stricter setup still would
# create these secrets OUT-OF-BAND (e.g. via the AWS CLI once) and have this
# config reference them by ARN via a data source, so the plaintext value
# never flows through `terraform plan`/state at all -- not done here to keep
# this a single, self-contained `terraform apply`, but worth calling out as
# the harder-but-safer alternative.
variable "anthropic_api_key" {
  description = "Anthropic API key for research_agent.py's LLM calls."
  type        = string
  sensitive   = true
  default     = ""
}

variable "neo4j_password" {
  description = "Password for the neo4j user (NEO4J_AUTH=neo4j/<this>)."
  type        = string
  sensitive   = true
  default     = "password" # local-dev value from docker-compose.yml -- override in production
}

variable "openai_api_key" {
  # Used by embeddings.py (text-embedding-3-small) -- both the backend, at
  # query time (research_agent.py's search tools call embed_query()), and
  # the daily-delta-job ETL task, at ingest time. The backend task
  # definition below did not previously pass this at all -- a gap from the
  # embedding engine's migration off local FastEmbed never having been
  # propagated to this Terraform config, meaning a deployed AWS backend
  # would fail every search today (embed_query() raises SystemExit with no
  # key set). Fixed here alongside adding the new ETL task, since both
  # need the same secret.
  description = "OpenAI API key for embeddings.py (text-embedding-3-small)."
  type        = string
  sensitive   = true
  default     = ""
}

variable "jwt_secret_key" {
  # Signs/verifies the Bearer JWTs auth.py validates on api.py's protected
  # endpoints -- same known gap as openai_api_key above: the backend task
  # definition never had this wired in, so a deployed AWS backend would
  # fail closed on every request (auth.py's get_current_user() raises 500
  # "JWT_SECRET_KEY is not configured" with no key set). Generate a real
  # value the same way the local .env one was generated -- `python -c
  # "import secrets; print(secrets.token_urlsafe(48))"` -- never a
  # human-chosen string; this default is intentionally empty so a
  # forgotten override fails loudly (auth.py's own 500) rather than
  # deploying with a guessable secret.
  description = "Signing key for auth.py's JWT validation (JWT_SECRET_KEY)."
  type        = string
  sensitive   = true
  default     = ""
}

variable "jwt_algorithm" {
  description = "JWT signing algorithm -- must match auth.py's JWT_ALGORITHM (default HS256)."
  type        = string
  default     = "HS256"
}

# research_agent.py's build_llm() routes agent orchestration/routing calls
# (tool-calling, intent classification, narrative synthesis) to whichever
# provider LLM_PROVIDER names -- "anthropic" (default, uses
# anthropic_api_key), "kimi", or "nvidia". The gpt-4o-pinned structured-
# extraction stage (Smart Table's Map workers, Landscape, Catalyst) always
# uses openai_api_key regardless of this setting -- see
# _build_gpt4o_llm's own docstring for why that stage specifically can't
# use a reasoning model or a low-concurrency-quota provider.
variable "llm_provider" {
  description = "Which LLM provider handles agent orchestration/routing: anthropic, kimi, or nvidia."
  type        = string
  default     = "anthropic"
}

variable "kimi_api_key" {
  description = "Moonshot (Kimi) API key for research_agent.py's build_llm() when llm_provider=kimi."
  type        = string
  sensitive   = true
  default     = ""
}

# --- service scaling -----------------------------------------------------------
# All default to 1. Qdrant/Neo4j CANNOT scale beyond 1 -- both are
# single-writer databases backed by one EFS volume, so a second task would
# corrupt or race against the first, not add capacity. backend/frontend
# variables exist so they COULD scale independently later; left at 1 by
# default to match the current single-instance local-dev deployment.
variable "qdrant_desired_count" {
  type    = number
  default = 1
}

variable "neo4j_desired_count" {
  type    = number
  default = 1
}

variable "backend_desired_count" {
  type    = number
  default = 1
}

variable "frontend_desired_count" {
  type    = number
  default = 1
}

variable "jobs_db_password" {
  # Same fail-loud philosophy as jwt_secret_key: empty default means a
  # forgotten override fails at RDS creation rather than shipping a
  # guessable password. Generate with:
  #   python -c "import secrets; print(secrets.token_urlsafe(24))"
  description = "Master password for the jobs Postgres (RDS) instance."
  type        = string
  sensitive   = true
  default     = ""
}
