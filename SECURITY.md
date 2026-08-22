# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Instead, use
GitHub's private vulnerability reporting on this repository
(Security → Report a vulnerability), or email the maintainer address on
the GitHub profile. You'll get an acknowledgement within a few days.

## Scope notes for self-hosters

- The reference deployment ships **without authentication** — every API
  endpoint is open. Do not expose an instance holding anything sensitive
  to the public internet without putting your own auth (ALB OIDC,
  reverse-proxy basic auth, VPN) in front of it. `auth.py` contains a JWT
  validator awaiting a real login flow; contributions welcome.
- API keys (OpenAI/Anthropic/Kimi) live in `.env` locally and AWS Secrets
  Manager in the Terraform deployment — never commit `.env`.
- The AACT text-to-SQL tool executes LLM-generated SQL behind a hard
  guard (single bare `SELECT`, keyword denylist, read-only transaction,
  statement timeout). If you find a bypass, that's exactly the kind of
  report we want.

## Supported versions

This is a fast-moving pre-1.0 project: only the latest `main` is
supported. Dependency versions are fully pinned via `uv.lock` /
`requirements-etl.lock.txt`, so a fresh clone builds the same bits we run.
