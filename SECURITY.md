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

Use GitHub's private vulnerability reporting form when it is available:
<https://github.com/muhammadmahadazher/laserperception/security/advisories/new>.

If GitHub does not present a private reporting option, open a public issue that requests a private
contact channel without including vulnerability details. Continue privately once the maintainer
provides that channel.

Include the affected version or commit, impact, minimal reproduction, and any suggested mitigation.
Remove credentials, private paths, proprietary data, datasets, checkpoints, and model artifacts.

Security support covers LaserPerception's own code. Vulnerabilities in optional upstream runtimes,
models, datasets, CUDA/driver stacks, ROS distributions, and cloud providers should also be
reported to their maintainers under their policies.
