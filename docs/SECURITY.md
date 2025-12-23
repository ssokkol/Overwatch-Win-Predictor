# Security Documentation

## Security Measures

### 1. Authentication

- **API Key Authentication**: All protected endpoints require valid API keys
- **Secret Management**: API keys stored as environment variables using Pydantic SecretStr
- **No Secret Logging**: Sensitive data is automatically redacted in logs

### 2. Input Validation

- **Strict Validation**: All inputs validated using Pydantic schemas
- **Type Checking**: Full type annotations and mypy checking
- **Range Validation**: Hero IDs validated against valid ranges
- **Duplicate Detection**: Prevents duplicate heroes in teams
- **Cross-Team Validation**: Ensures no hero appears in both teams

### 3. Rate Limiting

- **Per-IP Limiting**: 100 requests per 60 seconds (configurable)
- **Redis Backend**: Distributed rate limiting with Redis
- **Memory Fallback**: In-memory cache if Redis unavailable
- **Header Information**: Rate limit status in response headers

### 4. Security Headers

All responses include security headers:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `Content-Security-Policy: default-src 'self'`

### 5. CORS Configuration

- Configurable allowed origins
- Credentials support
- Restrictive default settings

### 6. Dependency Security

- **Vulnerability Scanning**: Safety and Dependabot for dependency checks
- **Pinned Versions**: Exact version pinning in requirements.txt
- **Regular Updates**: Scheduled security scans

### 7. Docker Security

- **Non-root User**: Application runs as non-root user
- **Minimal Base Image**: Python slim image
- **Read-only Filesystem**: Where possible
- **Capability Dropping**: Minimal required capabilities
- **Health Checks**: Container health monitoring

## Vulnerability Reporting

**Please DO NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: security@yourproject.com

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide a detailed response within 7 days.

## Security Checklist

### For Developers

- [ ] Never commit API keys or secrets
- [ ] Use `.env` files for local development
- [ ] Run security scans before committing
- [ ] Keep dependencies updated
- [ ] Follow secure coding practices
- [ ] Review code for injection vulnerabilities
- [ ] Validate all inputs
- [ ] Use parameterized queries (if using database)

### For Deployment

- [ ] Use strong API keys (random, long strings)
- [ ] Enable HTTPS in production
- [ ] Configure proper CORS origins
- [ ] Set up rate limiting
- [ ] Enable Redis for distributed caching
- [ ] Use secrets management (Vault, etc.)
- [ ] Regular security updates
- [ ] Monitor logs for suspicious activity

## Common Security Issues

### SQL Injection
- **Not Applicable**: No SQL database used

### XSS (Cross-Site Scripting)
- **Mitigation**: Security headers, input validation

### CSRF (Cross-Site Request Forgery)
- **Mitigation**: API key authentication, CORS configuration

### DoS (Denial of Service)
- **Mitigation**: Rate limiting, input validation

### Information Disclosure
- **Mitigation**: Secure logging, error message sanitization

## Compliance

This project follows:
- OWASP Top 10 best practices
- Python security best practices
- Docker security best practices
- API security guidelines

