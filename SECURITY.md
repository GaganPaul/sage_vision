# Security Policy

## Supported Versions

We actively support the following versions of **Sage Vision** with security updates:

| Version | Supported |
|--------|-----------|
| 1.x.x  | Yes |
| < 1.0  | No |

Only the latest stable release receives active security updates.

---

## Reporting a Vulnerability

We take security vulnerabilities seriously.  
If you discover a security issue in **Sage Vision**, please report it responsibly.

### Do NOT
- Do not report security vulnerabilities through **public GitHub issues**

---

## How to Report

Please report security vulnerabilities using one of the following methods:

- **GitHub Security Advisories**  
  Use GitHub’s private vulnerability reporting feature (preferred)

- **Direct Contact**  
  Contact the project maintainer directly through GitHub

*(If needed, open a private issue to request secure contact details.)*

---

## What to Include

When reporting a vulnerability, please provide:

- **Description** – Clear explanation of the issue  
- **Impact** – Potential severity and affected components  
- **Reproduction** – Steps to reproduce the vulnerability  
- **Environment** – Affected versions and configurations  
- **Mitigation** – Any temporary workarounds identified  

---

## Response Timeline

We aim to respond in a timely and transparent manner:

- **Acknowledgment** – Within 48 hours  
- **Initial Assessment** – Within 5 business days  
- **Status Updates** – Every 5 business days (if ongoing)  
- **Resolution Target** – Critical issues within 30 days  

---

## Security Best Practices

### For Developers

- **Keep Updated** – Always use the latest version  
- **Secure Configuration** – Follow recommended security settings  
- **Secrets Management** – Never commit API keys or secrets  
- **Input Validation** – Validate and sanitize all inputs  
- **Network Security** – Use HTTPS for all communications  

---

### For Production Deployments

- **Environment Variables** – Store secrets securely  
- **Access Control** – Apply authentication and authorization  
- **Monitoring** – Watch for abnormal behavior  
- **Logging** – Enable logs without exposing sensitive data  
- **Updates** – Regularly apply security patches  

---

## Scope

This security policy applies to:

- **Core Package** – Sage Vision core modules  
- **Pipelines & Models** – Built-in vision pipelines and AI models  
- **Dependencies** – Direct runtime dependencies  
- **Documentation** – Security-related documentation  
- **Examples** – Security issues in example code  

---

## Out of Scope

The following are generally out of scope:

- Third-party AI services or APIs  
- Issues caused by insecure user configuration  
- Development-only dependencies  
- Social engineering attacks  

---

## Recognition

We appreciate security researchers and contributors who help improve Sage Vision.

With your permission, we may acknowledge you in:

- Security advisories  
- Release notes  
- Project contributors list  

---

## Contact

For security-related concerns, please use the reporting channels mentioned above.

---

**Last Updated:** February 2026
