# Security Policy

This is a research code base for a university course project; only the latest commit on
`main` is maintained.

## Reporting a vulnerability

Please do not open a public issue. Email one of the authors listed in `pyproject.toml` at
their Columbia address with a description, the affected files and steps to reproduce. We aim
to respond within a week.

## Scope notes

- The harness executes model-generated code in a subprocess with a scrubbed environment and
  timeouts. That is not a security sandbox: run it only in a disposable environment such as
  Colab or a container.
- Endpoint credentials belong in `configs/vllm.env`, which is gitignored. Never commit keys.
