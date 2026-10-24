# Security Policy

## Supported Versions
Only the latest release on the main branch receives security patches.

| Version | Supported |
| :--- | :--- |
| 0.1.x (Alpha) | Yes |
| < 0.1.0 | No |

## Reporting a Vulnerability
Do NOT open public GitHub issues for security vulnerabilities.
Email vulnerability reports with steps to reproduce to: security@forgeoutlier.local (or create a private GitHub Security Advisory).
Reports will be acknowledged within 48 hours.

## Technical Security Controls
1. **Deserialization Boundaries**:
   - PyTorch: `weights_only=True` enforced unconditionally.
   - NumPy: `allow_pickle=False` enforced unconditionally.
2. **Resource Boundaries**:
   - Image maximum size: $50$ MB.
   - Image maximum dimensions: $4096 \times 4096$ pixels.
   - Video maximum frames: $1,000$ frames.
3. **Network Isolation**:
   - Zero runtime outbound HTTP/DNS socket connections.
