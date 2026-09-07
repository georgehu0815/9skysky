---
title: Configure npm and pnpm to use the Microsoft Node package proxy
description: Configure and verify a shared user npm registry setup for Microsoft's corporate Node package proxy on macOS and Linux.
author: George Hu
ms.date: 2026-09-05
ms.topic: how-to
---

## Why this configuration is needed

Node.js libraries are distributed as npm packages. Both
[npm](https://docs.npmjs.com/about-npm) and
[pnpm](https://pnpm.io/pnpm-cli) use the npm registry protocol and normally
read the same user `.npmrc`. The setup script therefore supports both package
managers through one user configuration file. It does not install Node.js,
npm, or pnpm.

On the managed Mac used to verify this setup, package metadata for `lodash`
resolved through `https://packagefeedproxy.microsoft.io/npm/`. A direct request
to the public npm registry failed with `ENOTCONN`. npm 11.11.0, Node.js
v22.22.0, and pnpm were installed, and both package managers resolved through
the Microsoft endpoint.

The verified endpoint is a Microsoft corporate dependency proxy. It is
separate from a team-owned Azure Artifacts feed. The client has one registry
and no public, scoped, or command-line fallback. Public upstream retrieval and
caching happen server-side. If the corporate proxy cannot serve a package, the
client fails closed instead of contacting `registry.npmjs.org`.

> [!IMPORTANT]
> The setup does not configure credentials, migrate tokens, weaken Microsoft
> Defender, or add a client-side public registry fallback.

This configuration applies only to the Node.js package ecosystem. Python and
uv use a separate configuration described in the
[Microsoft Python proxy guide](microsoft-python-proxy-guide.md).

## Prerequisites

* macOS or Linux with Bash
* [Node.js](https://nodejs.org/en/download) and
  [npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) for
  `--manager npm` or `--manager all`
* [pnpm](https://pnpm.io/installation) for `--manager pnpm`; pnpm is optional
  for `--manager all`
* Network access to `packagefeedproxy.microsoft.io` and Microsoft-hosted
  `*.pkgs.visualstudio.com` artifact endpoints

The script is available at
[setup-microsoft-node-proxy.sh](../scripts/setup-microsoft-node-proxy.sh).
It configures installed tools but never installs them.

## Install the shared user configuration

Run the script from the repository or use its absolute path:

```bash
scripts/setup-microsoft-node-proxy.sh install
```

The `install` command is the default, so this command is equivalent:

```bash
scripts/setup-microsoft-node-proxy.sh
```

By default, the shared user configuration path is:

```text
$HOME/.npmrc
```

If `NPM_CONFIG_USERCONFIG` is set, the script manages that absolute path
instead. Relative override paths are rejected. For example:

```bash
NPM_CONFIG_USERCONFIG="$HOME/config files/corporate.npmrc" \
  scripts/setup-microsoft-node-proxy.sh install
```

The backup state file is stored at:

```text
${XDG_CONFIG_HOME:-$HOME/.config}/npm/.microsoft-node-proxy-backup
```

Installation creates missing parent directories with restrictive permissions,
writes the npmrc atomically in its own directory, and sets mode `600`. Repeated
installation is idempotent when the file is the exact managed configuration or
the accepted one-line equivalent.

A different existing regular file requires confirmation. Noninteractive use
requires `--yes`:

```bash
scripts/setup-microsoft-node-proxy.sh install --yes
```

Before replacement, the script creates a timestamped mode-`600` backup next to
the npmrc and records its path. Existing registry credentials and other npm
settings are preserved in that backup, but they are not copied into the new
single-registry file. Replacing the file can temporarily remove access to
private scoped registries until rollback. The script never prints, parses, or
migrates tokens.

## Select package managers

The default `--manager all` mode requires Node.js and npm. It validates pnpm
when pnpm is installed and reports a skip when it is absent.

Choose one package manager when needed:

```bash
scripts/setup-microsoft-node-proxy.sh install --manager npm
scripts/setup-microsoft-node-proxy.sh install --manager pnpm
```

The manager selection controls prerequisite and check behavior. Both choices
manage the same user npmrc because npm and pnpm normally inherit that file.

## Check package resolution

Run the complete validation with the default package:

```bash
scripts/setup-microsoft-node-proxy.sh check
```

The check performs these actions without printing configuration contents:

* Reports the selected Node.js, npm, and pnpm versions and the npmrc path
* Verifies the exact single-registry configuration
* Creates a fresh temporary directory with no project npmrc
* Runs `npm view` and a script-disabled `npm install --dry-run`
* Runs `pnpm view` when selected or available in `all` mode
* Rejects `registry.npmjs.org`, non-HTTPS URLs, and unapproved artifact hosts
* Accepts the configured proxy host and Microsoft-hosted
  `*.pkgs.visualstudio.com` artifact hosts

No `--registry` option is supplied during checks. npm and pnpm must inherit the
user npmrc. pnpm has no reliable install dry-run, so its check uses metadata
resolution through `pnpm view`.

Use any registry package specification supported by `npm view`, including a
name, version, or scoped package:

```bash
scripts/setup-microsoft-node-proxy.sh check \
  --package '@types/node@22.10.2'
```

The default is `lodash@4.18.1`.

## Use npm and pnpm normally

After installation, normal commands use the shared registry configuration:

```bash
npm view lodash version
npm install lodash
npm ci
pnpm view lodash version
pnpm add lodash
pnpm install
```

The user npmrc is a default, not an enforcement boundary. These sources can
change or supersede registry selection:

* A project or workspace `.npmrc`
* An npm or pnpm command with `--registry`
* The `NPM_CONFIG_REGISTRY` environment variable
* pnpm configuration that sets another registry or scoped registry
* A different `NPM_CONFIG_USERCONFIG` value

Use the official
[npm configuration documentation](https://docs.npmjs.com/cli/v11/configuring-npm/npmrc)
and [pnpm configuration documentation](https://pnpm.io/configuring) to inspect
precedence. Remove unintended overrides before rerunning `check`.

### Handle lockfiles deliberately

A `package-lock.json` can contain old public `resolved` URLs. pnpm lockfiles and
store metadata can also retain prior resolution information. Do not blindly
edit lockfiles or replace host strings. Regenerate a lockfile only after
reviewing the dependency change, confirming the intended registry, and
accepting the resulting version and integrity changes.

For npm, a deliberate regeneration might involve removing and recreating the
lockfile under the approved configuration. Follow the repository's dependency
update process and review the complete diff before committing it.

## Remove or roll back the configuration

Remove the setup with:

```bash
scripts/setup-microsoft-node-proxy.sh remove
```

If installation replaced a prior npmrc, removal restores that exact backup and
preserves the backup file. If no prior configuration was recorded, removal
deletes only the exact script-managed file. The state file contains one plain
backup path and is never sourced as shell code. Recorded paths are accepted
only when they use the managed npmrc backup prefix.

An unrecognized or user-edited file is refused unless `--yes` is supplied.
Forced removal creates a mode-`600` timestamped backup before deleting the
file:

```bash
scripts/setup-microsoft-node-proxy.sh remove --yes
```

Display concise command help with:

```bash
scripts/setup-microsoft-node-proxy.sh help
```

## Configure the file manually

The equivalent user npmrc contains one registry and no authentication or
fallback settings:

```ini
# Managed by setup-microsoft-node-proxy.sh.
# Microsoft corporate npm proxy. Keep this as the sole registry.
registry=https://packagefeedproxy.microsoft.io/npm/
# End setup-microsoft-node-proxy.sh managed configuration.
```

Set mode `600` on the file. `always-auth=false` is unnecessary for this
endpoint and is intentionally omitted. Do not add tokens, scoped public
fallbacks, or `registry.npmjs.org` as another source.

See the official [npm registry configuration](https://docs.npmjs.com/cli/v11/using-npm/registry)
for registry behavior.

## Troubleshoot common failures

### Direct public npm fails with ENOTCONN or Defender blocks access

`ENOTCONN` can indicate that managed network protection blocked the public
endpoint even though DNS or some metadata requests succeeded. Confirm that the
user npmrc points only to the approved Microsoft proxy, then rerun `check`.
Do not disable or bypass
[Microsoft Defender Network Protection](https://learn.microsoft.com/defender-endpoint/network-protection-macos).
Request an enterprise allow decision when an approved endpoint is blocked.

### pnpm is absent

`--manager all` reports that pnpm is skipped and still validates npm. Install
pnpm through the organization's approved process if it is required. The
`--manager pnpm` mode fails immediately when pnpm is unavailable.

### The proxy root behaves differently from a package request

A registry root request can return a status, redirect, or response that differs
from package metadata and tarball requests. Test the package operation that the
client needs:

```bash
npm view lodash@4.18.1 dist.tarball
pnpm view lodash@4.18.1 dist.tarball
```

Avoid adding a public fallback because a root listing is unavailable. The
script validates the resolved tarball host instead.

### TLS validation fails

Keep TLS verification enabled. Check the system clock, managed trust store,
corporate certificate deployment, and proxy service health. Do not set
`strict-ssl=false` or place credentials in the registry URL. npm documents its
TLS-related settings in the
[npm configuration reference](https://docs.npmjs.com/cli/v11/using-npm/config).

### A cached package works while a new package fails

npm and pnpm caches can satisfy an operation without a network request. A
successful cached install does not prove that the proxy is reachable. Use the
script's metadata check with an approved package and review cache behavior in
the [npm cache documentation](https://docs.npmjs.com/cli/v11/commands/npm-cache)
or [pnpm store documentation](https://pnpm.io/cli/store).

Offline modes fail when required metadata or artifacts are absent. Prepare and
validate caches intentionally before relying on offline installation.

### The check detects a public or unapproved host

Inspect project npmrc files, `NPM_CONFIG_REGISTRY`, CLI aliases, pnpm settings,
and lockfiles. Remove the override rather than weakening the host check. The
script does not display captured command output because registry URLs can carry
sensitive query data.

## Consider Azure Artifacts for enterprise ownership

Teams that need an owned feed, explicit permissions, retention, promotion, or
audited upstream-source policy should use
[Azure Artifacts for npm](https://learn.microsoft.com/azure/devops/artifacts/npm/npmrc)
and configure
[upstream sources](https://learn.microsoft.com/azure/devops/artifacts/concepts/upstream-sources).
That design is a team-owned enterprise feed and may require authentication.
Keep it separate from this corporate proxy configuration instead of adding it
as a fallback registry.
