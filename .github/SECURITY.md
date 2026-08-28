# Security policy

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report it privately through GitHub's private vulnerability reporting:
**[Security → Report a vulnerability](https://github.com/MemoryUniverse/mu-core/security/advisories/new)**.
That channel is visible only to you and the maintainers, and it gives us a place to coordinate a fix
and a disclosure date with you.

If the form is not available to you for any reason, open a normal issue that says only *"I need a
private channel for a security report"* — with **no details of the problem in it** — and a
maintainer will open the advisory and invite you.

## What to include

- What an attacker can do, and what they need in order to do it.
- The smallest reproduction you have. A failing test is ideal.
- The commit or branch you saw it on.

**Do not include real memory content, credentials, or personal data in a report.** If a reproduction
seems to need them, redact and tell us what you redacted — this project's whole discipline is that
remembered text never leaves the machine it was remembered on, and a security report is not the
place to break it.

## What to expect

This is an early-stage project maintained by a small team, so plain numbers rather than a promise
we would have to break:

| | Target |
|---|---|
| Acknowledgement | within 3 working days |
| First assessment (is it real, how bad) | within 10 working days |
| Fix or a dated plan | agreed with you on the advisory |

We will credit you in the advisory unless you ask us not to.

## Supported versions

**None yet.** There is no released version of `mu-core`: no git tag, no package on PyPI. The only
supported thing is the tip of the integration trunk. Fixes land there and nowhere else until the
first release, which is described in [RELEASING.md](RELEASING.md).

## Scope

In scope — anything in this repository, and in particular:

- A path where memory content, a credential, or a namespace's data reaches a log, trace, event,
  metric or error message.
- A namespace-isolation failure: any way one namespace (org / workspace / user / agent / session)
  reads or writes another's data.
- An authorization check that can be bypassed on a read or write path.
- Injection, deserialization or path-traversal reachable from stored memory content, since that
  content is attacker-influenced by design.

Out of scope — vulnerabilities in third-party dependencies with no exploitable path through this
code (report those upstream, and tell us so we can pin), findings that require an already-root
attacker on the machine, and anything about the hosted plane (`mu-server`), which is not in this
repository.
