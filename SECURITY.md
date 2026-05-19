# Security Policy

## Supported Versions

Only the latest commit on the `main` branch receives security fixes.

| Version | Supported |
|---------|-----------|
| `main` (latest) | Yes |
| Older commits | No |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Use GitHub's private vulnerability reporting feature instead:

1. Go to the [Security tab](../../security) of this repository.
2. Click **"Report a vulnerability"**.
3. Fill in the details and submit.

This keeps the report confidential until a fix is available.

## Response Timeline

| Milestone | Target |
|-----------|--------|
| Acknowledgement | Within 48 hours |
| Fix or mitigation | Within 7 days for critical issues |
| Public disclosure | After fix is released |

## Scope

This project is a local reverse proxy for routing API requests. The primary
security concerns are:

- Credential handling (`credentials.json`)
- Token forwarding and header injection
- Local network exposure (default: `127.0.0.1` only)

If you discover an issue outside this scope, please still report it — we will
triage and respond accordingly.
