# Security policy

## Supported versions

Security fixes are applied to the current release line and `main`. v0.4.0 is the current supported
release. Historical v0.1.0–v0.3.0 tags and their frozen evidence remain available for
reproducibility but do not receive routine backports.

| Version | Supported |
|---|---|
| 0.4.x | Yes |
| 0.3.x and earlier | No routine backports |

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's private vulnerability
reporting for this repository when available, or contact the maintainer through the private
security contact shown on the repository's Security page.

Include the affected version or commit, impact, minimal reproduction, and any suggested mitigation.
Remove credentials, private paths, proprietary data, datasets, checkpoints, and model artifacts.

Security support covers LaserPerception's own code. Vulnerabilities in optional upstream runtimes,
models, datasets, CUDA/driver stacks, ROS distributions, and cloud providers should also be
reported to their maintainers under their policies.
