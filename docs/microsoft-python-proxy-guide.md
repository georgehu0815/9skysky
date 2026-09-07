---
title: Configure uv to use the Microsoft Python proxy
description: Configure and verify a global single-index uv setup for Microsoft's corporate Python package proxy on macOS and Linux.
author: George Hu
ms.date: 2026-09-05
ms.topic: how-to
---

## Why this configuration is needed

On the managed Mac used to verify this setup, Microsoft Defender Network
Protection blocks `files.pythonhosted.org` through a managed custom block list.
Name resolution and package metadata from `pypi.org` can still work, but direct
artifact downloads fail. This split behavior can make a normal uv resolution
look healthy until the wheel or source archive is downloaded.

The corporate proxy at
`https://packagefeedproxy.microsoft.io/pypi/simple/` provides the Python simple
index and returns package artifacts from Microsoft-hosted
`*.pkgs.visualstudio.com` endpoints. Upstream retrieval and caching happen
server-side in Microsoft's service. uv is configured with one index only and
does not perform an automatic client-side fallback to public PyPI. If the proxy
is unavailable or cannot serve a package, uv fails closed instead of contacting
`pypi.org` or `files.pythonhosted.org`.

This service is a Microsoft corporate dependency proxy. It is not a public
Azure Artifacts feed owned or administered by this repository.

> [!IMPORTANT]
> The setup does not configure credentials, weaken Defender, or add PyPI or an
> unofficial source as a fallback index.

## Prerequisites

* macOS or Linux with Bash
* [uv installed](https://docs.astral.sh/uv/getting-started/installation/)
* A Python interpreter discoverable by uv or available as `python3` or `python`
* Network access to `packagefeedproxy.microsoft.io` and the Microsoft-hosted
  artifact endpoints returned by the proxy
* Optional `curl` for package-page artifact-host validation

The setup follows uv's
[configuration-file rules](https://docs.astral.sh/uv/concepts/configuration-files/)
and uses the modern
[index configuration](https://docs.astral.sh/uv/concepts/indexes/).

## Install the global configuration

Run the repository script from any directory:

```bash
/path/to/microduck-lab/scripts/setup-microsoft-python-proxy.sh install
```

The script is available at
[setup-microsoft-python-proxy.sh](../scripts/setup-microsoft-python-proxy.sh).
The `install` command is the default, so omitting it has the same effect.

The global path is based on the XDG configuration convention:

```text
${XDG_CONFIG_HOME:-$HOME/.config}/uv/uv.toml
```

With the default macOS and Linux environment, this resolves to:

```text
$HOME/.config/uv/uv.toml
```

Installation creates the directory when needed and protects the configuration
with mode `600`. A replacement is written through a temporary file in the same
directory and moved atomically. If a different configuration already exists,
the script prompts before replacement, creates a timestamped backup, and records
the exact backup path in the uv configuration directory. Noninteractive
replacement requires `--yes`.

An existing exact managed configuration is left unchanged, which makes repeated
installation safe:

```bash
scripts/setup-microsoft-python-proxy.sh
scripts/setup-microsoft-python-proxy.sh install
scripts/setup-microsoft-python-proxy.sh install --yes
```

Use `--url` only for another approved instance of the same service. The script
requires HTTPS and rejects credentials embedded in a URL:

```bash
scripts/setup-microsoft-python-proxy.sh install \
  --url https://packagefeedproxy.microsoft.io/pypi/simple/
```

## Check package resolution

Run the complete check with the default validation package:

```bash
scripts/setup-microsoft-python-proxy.sh check
```

The check reports the uv version and global configuration path, verifies the
expected configuration exactly, detects Python without requiring pip, and runs
the following operation from a new temporary directory:

```bash
uv pip install --dry-run --target TEMP_DIRECTORY \
  --python DETECTED_PYTHON tensorboard-data-server==0.7.2
```

No index flags are supplied. The verbose output must reference the configured
Microsoft proxy and must not reference `pypi.org` or
`files.pythonhosted.org`. When `curl` is present, the check also reads the
package page, reports discovered artifact hosts, and accepts only the configured
proxy host or `*.pkgs.visualstudio.com`.

Select a different package when necessary:

```bash
scripts/setup-microsoft-python-proxy.sh check \
  --package 'requests==2.32.5'
```

## Use uv normally from any project

After installation, normal uv commands launched from any directory use the
global index:

```bash
uv sync
uv add requests
uv pip install numpy
uv run python -c 'import sys; print(sys.version)'
```

The global file is a default, not an enforcement boundary. Project-local uv
configuration, command-line index options, and uv environment variables can
override or augment global settings. If the check detects a public host, inspect
`pyproject.toml`, `uv.toml`, command aliases, and variables such as
`UV_INDEX`, `UV_DEFAULT_INDEX`, and legacy `UV_INDEX_URL`. See the official
[uv environment variable reference](https://docs.astral.sh/uv/reference/environment/)
for the current list.

## Understand the pip distinction

This configuration controls uv. It does not rewrite pip configuration files,
including system, user, or virtual-environment files such as `pip.conf` and
`pip.ini`. Existing system pip configuration may already point to a corporate
index, but uv's native index configuration remains separate.

Use `uv pip` when you need pip-compatible operations under uv's resolver. Avoid
adding direct public PyPI fallback settings to either client on a managed
network.

## Remove or roll back the configuration

Remove the managed setup with:

```bash
scripts/setup-microsoft-python-proxy.sh remove
```

If installation replaced a prior configuration, removal restores that exact
backup and preserves the backup file. If no prior configuration was recorded,
removal deletes only the exact script-managed file. An unrecognized or
user-edited current file is refused unless `--yes` is supplied. Forced removal
backs up that file first.

```bash
scripts/setup-microsoft-python-proxy.sh remove --yes
```

The script stores only a plain backup path as state. It never sources the state
file as shell code. Backups remain next to the global uv configuration with
names like:

```text
uv.toml.backup.20260905T120000Z
```

## Configure the file manually

To apply the equivalent configuration manually, create the global uv file with
mode `600` and no additional indexes:

```toml
# Managed by setup-microsoft-python-proxy.sh.
# Microsoft corporate PyPI proxy. It retrieves and caches public PyPI packages.
# Keep this as the sole index to avoid dependency-confusion across repositories.
[[index]]
name = "microsoft"
url = "https://packagefeedproxy.microsoft.io/pypi/simple/"
default = true
# End setup-microsoft-python-proxy.sh managed configuration.
```

For details about index priority and `default = true`, consult the official
[uv index documentation](https://docs.astral.sh/uv/concepts/indexes/).

## Troubleshoot common failures

### Virtual environment mismatch

A warning that `VIRTUAL_ENV` does not match a project's expected environment is
separate from proxy resolution. Leave the environment, unset `VIRTUAL_ENV`, or
select the intended project environment before running the original uv command.
Do not change index configuration to address that warning.

### Cross-filesystem link warning

uv may report that hardlinking failed when its cache and target environment are
on different filesystems. Set `UV_LINK_MODE=copy` when copying is acceptable:

```bash
UV_LINK_MODE=copy uv sync
```

This warning concerns artifact installation strategy, not proxy access.

### Root listing returns 401

A request to the proxy root can return HTTP 401 while a package-specific simple
index page returns HTTP 200. Test the package page that uv actually needs:

```bash
curl --fail --location \
  https://packagefeedproxy.microsoft.io/pypi/simple/tensorboard-data-server/
```

The setup script checks the package-specific page rather than treating root
listing access as a prerequisite.

### Check reports a public host

Project configuration, CLI flags, or environment variables can supersede the
global file. Remove the override and rerun `check`. The script intentionally
fails when verbose uv output references `pypi.org` or
`files.pythonhosted.org`.

### Proxy or package is unavailable

uv fails when the sole proxy cannot serve the requested package. It does not
reach public PyPI automatically. Confirm corporate network access, package name,
version availability, TLS trust, and service health. Do not bypass managed
[Microsoft Defender Network Protection](https://learn.microsoft.com/defender-endpoint/network-protection-macos)
or add an unofficial fallback.

## Work with caches and offline mode

Successful proxy downloads populate uv's cache. A later operation may succeed
without network access when every required artifact and item of metadata is
already cached. Cache presence is not guaranteed, and a lockfile alone does not
contain package artifacts.

Use uv's documented offline mode only when the cache is intentionally prepared:

```bash
uv sync --offline
```

An offline cache miss fails. uv does not contact the proxy or public PyPI while
offline. Review the official
[uv cache documentation](https://docs.astral.sh/uv/concepts/cache/) before
building an offline workflow.

## Consider Azure Artifacts for enterprise ownership

Teams that need an owned feed, explicit permissions, retention policy, package
promotion, or audited upstream-source policy should use
[Azure Artifacts for Python](https://learn.microsoft.com/azure/devops/artifacts/python/project-setup-python).
An enterprise feed can centralize governance and credentials. Configure it as a
separate approved design rather than adding it as an opportunistic fallback to
this single-index setup.
