# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please DO NOT report security vulnerabilities through public GitHub issues.**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide a detailed response within 7 days.

## Security Measures

- All dependencies are regularly scanned with Safety and Dependabot
- Docker images are scanned with Trivy
- Code is analyzed with Bandit for security issues
- API rate limiting is enforced
- All inputs are validated with Pydantic
- Secrets are managed via environment variables
- Security headers are added to all responses

