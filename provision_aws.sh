#!/usr/bin/env bash
#
# provision_aws.sh -- create and harden the raw data lake bucket via AWS CLI.
#
# Idempotent: every step is safe to re-run. Reads S3_BUCKET / AWS_DEFAULT_REGION
# from .env unless they are already exported.
#
#   ./provision_aws.sh            # provision
#   ./provision_aws.sh --dry-run  # print what would happen, change nothing
#
set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:$PATH"

# --- load .env (only the keys we need; ignores comments/blank lines) ---------
if [[ -f .env ]]; then
  while IFS='=' read -r k v; do
    [[ "$k" =~ ^[A-Z0-9_]+$ ]] || continue
    [[ -n "${v:-}" ]] || continue
    case "$k" in
      S3_BUCKET|AWS_DEFAULT_REGION|AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN)
        [[ -z "${!k:-}" ]] && export "$k=$v" ;;
    esac
  done < <(grep -E '^[A-Z0-9_]+=' .env)
fi

BUCKET="${S3_BUCKET:?S3_BUCKET not set in .env}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"

say()  { printf '\033[1m[provision]\033[0m %s\n' "$*"; }
run()  { if $DRY_RUN; then echo "  DRY-RUN: aws $*"; else aws "$@"; fi; }

echo "======================================================================"
echo " AWS provisioning :: s3://$BUCKET  ($REGION)"
$DRY_RUN && echo " MODE: DRY RUN -- nothing will be created"
echo "======================================================================"

# --- 0. preflight: who am I? -------------------------------------------------
say "checking credentials ..."
if ! IDENTITY=$(aws sts get-caller-identity --output json 2>&1); then
  cat >&2 <<EOF

[provision] FAILED: no usable AWS credentials.
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

# --- 1. bucket ---------------------------------------------------------------
if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  say "bucket already exists and is accessible -- skipping create"
else
  say "creating bucket $BUCKET in $REGION ..."
  # us-east-1 is the API default and REJECTS an explicit LocationConstraint.
  if [[ "$REGION" == "us-east-1" ]]; then
    run s3api create-bucket --bucket "$BUCKET" --region "$REGION"
  else
    run s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
        --create-bucket-configuration "LocationConstraint=$REGION"
  fi
  $DRY_RUN || aws s3api wait bucket-exists --bucket "$BUCKET"
  say "bucket created"
fi

# --- 2. block ALL public access ---------------------------------------------
# Clinical trial data is public, but a data lake should never be world-readable.
say "blocking public access ..."
run s3api put-public-access-block --bucket "$BUCKET" \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

# --- 3. versioning -----------------------------------------------------------
# Raw payloads are the replay source of truth; versioning makes an accidental
# overwrite recoverable.
say "enabling versioning ..."
run s3api put-bucket-versioning --bucket "$BUCKET" \
  --versioning-configuration Status=Enabled

# --- 4. default encryption ---------------------------------------------------
say "enabling default encryption (SSE-S3/AES256) ..."
run s3api put-bucket-encryption --bucket "$BUCKET" \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"},"BucketKeyEnabled":true}]}'

# --- 5. lifecycle ------------------------------------------------------------
# Raw JSON is written once and read rarely -- tier it down instead of paying
# Standard forever, and reap old versions so versioning does not grow unbounded.
say "applying lifecycle policy ..."
run s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" \
  --lifecycle-configuration '{
    "Rules": [{
      "ID": "raw-tiering",
      "Status": "Enabled",
      "Filter": {"Prefix": "raw/"},
      "Transitions": [
        {"Days": 30, "StorageClass": "STANDARD_IA"},
        {"Days": 90, "StorageClass": "GLACIER_IR"}
      ],
      "NoncurrentVersionExpiration": {"NoncurrentDays": 180},
      "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7}
    }]
  }'

# --- 6. tags -----------------------------------------------------------------
say "tagging ..."
run s3api put-bucket-tagging --bucket "$BUCKET" \
  --tagging 'TagSet=[{Key=project,Value=medical-rag},{Key=layer,Value=raw},{Key=managed-by,Value=provision_aws.sh}]'

if $DRY_RUN; then
  echo; say "DRY RUN complete -- nothing was created."
  exit 0
fi

# --- 7. verify by actually reading the config back --------------------------
echo
echo "----------------------------------------------------------------------"
echo " VERIFICATION (read back from AWS)"
echo "----------------------------------------------------------------------"
printf 'region        : %s\n' "$(aws s3api get-bucket-location --bucket "$BUCKET" --query 'LocationConstraint' --output text)"
printf 'versioning    : %s\n' "$(aws s3api get-bucket-versioning --bucket "$BUCKET" --query 'Status' --output text)"
printf 'encryption    : %s\n' "$(aws s3api get-bucket-encryption --bucket "$BUCKET" --query 'ServerSideEncryptionConfiguration.Rules[0].ApplyServerSideEncryptionByDefault.SSEAlgorithm' --output text)"
printf 'public access : %s\n' "$(aws s3api get-public-access-block --bucket "$BUCKET" --query 'PublicAccessBlockConfiguration.BlockPublicPolicy' --output text)"
printf 'lifecycle     : %s\n' "$(aws s3api get-bucket-lifecycle-configuration --bucket "$BUCKET" --query 'Rules[0].ID' --output text)"

# round-trip a real object so we prove write+read+delete, not just config
say "round-tripping a test object ..."
TMP=$(mktemp)
echo '{"provisioned":true}' > "$TMP"
aws s3 cp "$TMP" "s3://$BUCKET/_healthcheck/provision.json" --only-show-errors
aws s3 cp "s3://$BUCKET/_healthcheck/provision.json" - | head -1
aws s3 rm "s3://$BUCKET/_healthcheck/provision.json" --only-show-errors
rm -f "$TMP"
say "read/write/delete OK"

echo
say "provisioned: s3://$BUCKET"
