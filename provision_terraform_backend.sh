#!/usr/bin/env bash
#
# provision_terraform_backend.sh -- create the S3 bucket + DynamoDB lock
# table that terraform/main.tf's `backend "s3"` block points at.
#
# WHY THIS EXISTS: terraform/main.tf previously had no `backend` block at
# all -- state lived in a local terraform.tfstate file inside terraform/,
# fine for one engineer running `terraform apply` from their own laptop,
# but genuinely dangerous the moment a GitHub Actions runner also runs
# `terraform apply` (see .github/workflows/deploy.yml's Terraform Apply
# job): a fresh runner checkout has NO local state file, so Terraform
# would conclude nothing exists yet and attempt to CREATE A SECOND COPY of
# this stack (VPC, ECS cluster, ALB, EFS, ...) alongside the real one --
# at best wasted spend and duplicate resources, at worst name collisions
# (S3/ECR names are globally/account-unique) hard-failing the pipeline
# mid-apply. Remote state, shared between every `terraform apply` caller
# (a laptop or a CI runner), is what prevents that.
#
# This is a bootstrap chicken-and-egg problem BY DESIGN, not an oversight:
# Terraform needs its backend's own storage to already exist before
# `terraform init` can use it, so that storage cannot itself be created BY
# the same Terraform config -- hence a plain AWS CLI script, matching this
# repo's own provision_aws.sh pattern for the identical class of problem
# (the S3 data-lake bucket also isn't Terraform-managed, for the same
# reason).
#
# Idempotent: every step is safe to re-run.
#
#   ./provision_terraform_backend.sh              # provision
#   ./provision_terraform_backend.sh --dry-run     # print what would happen, change nothing
#
# AFTER this script succeeds, migrate the EXISTING local state (this
# stack is already live -- see terraform/terraform.tfstate) rather than
# starting fresh:
#
#   cd terraform
#   terraform init -migrate-state
#
# No -backend-config= flags needed: terraform/main.tf's `backend "s3"`
# block already has this script's exact bucket/table/region values
# hardcoded (not secret, so committing them is fine -- see that block's
# own comment for why this was chosen over a partial config).
#
# `-migrate-state` is what copies the CURRENT local state up to S3 instead
# of discarding it -- Terraform will ask for confirmation before doing so.
# This step is NOT run by this script or by any automation in this repo:
# it touches the state of real, already-applied infrastructure, so it is
# left as a deliberate, reviewed, one-time action for a human to run.

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:$PATH"

REGION="${AWS_DEFAULT_REGION:-us-east-1}"

say()  { printf '\033[1m[provision-tf-backend]\033[0m %s\n' "$*"; }
run()  { if $DRY_RUN; then echo "  DRY-RUN: aws $*"; else aws "$@"; fi; }

echo "======================================================================"
echo " Terraform remote-state backend provisioning ($REGION)"
$DRY_RUN && echo " MODE: DRY RUN -- nothing will be created"
echo "======================================================================"

# --- 0. preflight: who am I? -------------------------------------------------
say "checking credentials ..."
if ! IDENTITY=$(aws sts get-caller-identity --output json 2>&1); then
  cat >&2 <<EOF

[provision-tf-backend] FAILED: no usable AWS credentials.
                        $(echo "$IDENTITY" | tail -1)

  Authenticate first, then re-run this script:
      aws configure                  # long-lived access key
      aws configure sso              # IAM Identity Center / SSO

EOF
  exit 1
fi
ACCOUNT=$(echo "$IDENTITY" | grep -o '"Account": *"[^"]*"' | cut -d'"' -f4)
ARN=$(echo "$IDENTITY" | grep -o '"Arn": *"[^"]*"' | cut -d'"' -f4)
say "account : $ACCOUNT"
say "identity: $ARN"

# Suffixed with the account id: S3 bucket names are GLOBALLY unique across
# all AWS accounts, not just this one, so a plain "medical-rag-terraform-state"
# risks colliding with an unrelated bucket some other AWS customer already owns.
BUCKET="medical-rag-terraform-state-${ACCOUNT}"
LOCK_TABLE="medical-rag-terraform-locks"

# --- 1. state bucket ----------------------------------------------------------
if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  say "state bucket already exists and is accessible -- skipping create"
else
  say "creating state bucket $BUCKET in $REGION ..."
  if [[ "$REGION" == "us-east-1" ]]; then
    run s3api create-bucket --bucket "$BUCKET" --region "$REGION"
  else
    run s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
        --create-bucket-configuration "LocationConstraint=$REGION"
  fi
  $DRY_RUN || aws s3api wait bucket-exists --bucket "$BUCKET"
  say "state bucket created"
fi

say "blocking public access ..."
run s3api put-public-access-block --bucket "$BUCKET" \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

# Versioning on a STATE bucket specifically: a corrupted or bad `apply`
# writing broken state is recoverable by rolling back to the prior version,
# the same reason provision_aws.sh enables it on the raw data lake bucket.
say "enabling versioning ..."
run s3api put-bucket-versioning --bucket "$BUCKET" \
  --versioning-configuration Status=Enabled

say "enabling default encryption (SSE-S3/AES256) ..."
run s3api put-bucket-encryption --bucket "$BUCKET" \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"},"BucketKeyEnabled":true}]}'

say "tagging ..."
run s3api put-bucket-tagging --bucket "$BUCKET" \
  --tagging 'TagSet=[{Key=project,Value=medical-rag},{Key=layer,Value=terraform-state},{Key=managed-by,Value=provision_terraform_backend.sh}]'

# --- 2. lock table -------------------------------------------------------------
# DynamoDB, not S3-native locking (Terraform >=1.10's `use_lockfile`):
# picked for broad compatibility with whichever Terraform version ends up
# running this (a GitHub Actions runner's pinned version isn't controlled
# by this script), not because native locking is unavailable on the
# Terraform 1.15.x installed locally.
if aws dynamodb describe-table --table-name "$LOCK_TABLE" >/dev/null 2>&1; then
  say "lock table already exists -- skipping create"
else
  say "creating lock table $LOCK_TABLE ..."
  run dynamodb create-table \
    --table-name "$LOCK_TABLE" \
    --attribute-definitions AttributeName=LockID,AttributeType=S \
    --key-schema AttributeName=LockID,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --tags Key=project,Value=medical-rag Key=managed-by,Value=provision_terraform_backend.sh
  $DRY_RUN || aws dynamodb wait table-exists --table-name "$LOCK_TABLE"
  say "lock table created"
fi

if $DRY_RUN; then
  echo; say "DRY RUN complete -- nothing was created."
  exit 0
fi

echo
echo "----------------------------------------------------------------------"
echo " VERIFICATION (read back from AWS)"
echo "----------------------------------------------------------------------"
printf 'bucket versioning : %s\n' "$(aws s3api get-bucket-versioning --bucket "$BUCKET" --query 'Status' --output text)"
printf 'bucket encryption  : %s\n' "$(aws s3api get-bucket-encryption --bucket "$BUCKET" --query 'ServerSideEncryptionConfiguration.Rules[0].ApplyServerSideEncryptionByDefault.SSEAlgorithm' --output text)"
printf 'lock table status  : %s\n' "$(aws dynamodb describe-table --table-name "$LOCK_TABLE" --query 'Table.TableStatus' --output text)"

echo
say "provisioned: s3://$BUCKET  +  dynamodb:$LOCK_TABLE"
echo
say "NEXT STEP (run by hand, not by this script -- see this file's header):"
echo
echo "    Confirm terraform/main.tf's backend \"s3\" block has these exact values"
echo "    (bucket=$BUCKET, dynamodb_table=$LOCK_TABLE, region=$REGION), then:"
cat <<EOF

    cd terraform
    terraform init -migrate-state

EOF
