# Security

`ml-systems-lab` runs commands on machines you point it at: it copies an agent over
SSH, executes it with `kubectl exec`, and launches local inference binaries. A config
file is therefore code, and should be treated as such: run configs you wrote or have
read, the same way you would a shell script.

## Reporting a vulnerability

Please report security problems privately, through GitHub's
[private vulnerability reporting](https://github.com/manunicholasjacob/ml-systems-lab/security/advisories/new)
for this repository, rather than in a public issue. Include the version
(`mlsys --version`), the command or config, and what an attacker controls.

This is a single-maintainer project, so expect an acknowledgement within days rather
than hours. Fixes are released as a new PyPI version and noted in `CHANGELOG.md`.

## Supported versions

Only the latest release on PyPI receives fixes.
