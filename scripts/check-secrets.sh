#!/usr/bin/env bash
# check-secrets.sh
# Scans EVERY revision of this repository for committed AWS credentials.
# The TA runs exactly this before grading. Exit 0 = clean, exit 1 = findings.
#
# What it looks for, across all commits on all branches and tags:
#   - AWS access key IDs: AKIA followed by 16 uppercase letters/digits
#   - AWS secret access keys assigned in config: aws_secret_access_key = <40 chars>
#   - tracked terraform.tfvars, *.tfstate, or .env files
#
# What it deliberately ignores:
#   - AKIAIOSFODNN7EXAMPLE, the placeholder key AWS uses in its own docs. It is
#     what LocalStack returns from sts get-caller-identity, so it appears in
#     docs/lab1b-localstack-output.txt and environments/local/README.md in
#     every correct submission.
#   - The word "AKIA" on its own (prose such as "starts with AKIA").
#
# Usage: bash scripts/check-secrets.sh   (from the repo root)

set -uo pipefail

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a git repository."; exit 2
fi

revs=$(git rev-list --all)
findings=0

echo "== Access key IDs (AKIA + 16 chars), all revisions =="
hits=$(git grep -I -E -n -o 'AKIA[0-9A-Z]{16}' $revs 2>/dev/null \
  | grep -v 'AKIAIOSFODNN7EXAMPLE' || true)
if [ -n "$hits" ]; then echo "$hits"; findings=1; else echo "  none"; fi

echo "== Secret access keys assigned in config, all revisions =="
hits=$(git grep -I -E -n -i 'aws_secret_access_key[[:space:]]*[=:][[:space:]]*["'"'"']?[A-Za-z0-9/+]{40}' $revs 2>/dev/null || true)
if [ -n "$hits" ]; then echo "$hits"; findings=1; else echo "  none"; fi

echo "== Credential-bearing files ever tracked =="
hits=$(git log --all --name-only --pretty=format: 2>/dev/null \
  | grep -E '(^|/)(terraform\.tfvars|\.env)$|\.tfstate(\.backup)?$' | sort -u || true)
if [ -n "$hits" ]; then echo "$hits"; findings=1; else echo "  none"; fi

echo
if [ "$findings" -eq 0 ]; then
  echo "CLEAN: no credentials found in git history."
else
  echo "FINDINGS ABOVE. Rotate the key in the AWS console first, then remove it from history."
  exit 1
fi
