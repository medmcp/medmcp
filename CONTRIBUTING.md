# Contributing to MedMCP

Thank you for your interest in contributing to MedMCP!
This guide will help you get set up and familiar with our development workflow.

## Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

## Development setup

1. Fork and clone the repository:

   ```bash
   git clone https://github.com/<your-username>/medmcp.git
   cd medmcp
   ```

2. Install the project with all dev dependencies:

   ```bash
   uv sync
   ```

   This installs the package in editable mode together with the `test` and
   `quality` dependency groups (linting, type-checking, testing).

3. Install the pre-commit hooks:

   ```bash
   uv run pre-commit install
   ```

## Running checks locally

Before pushing, make sure all checks pass - these are the same checks that run
in CI:

```bash
uv run ruff check          # lint
uv run ruff format --check # formatting
uv run pyright             # type-checking (strict mode)
uv run pytest              # tests
```

To auto-fix lint and formatting issues:

```bash
uv run ruff check --fix
uv run ruff format
```

## Code style

- **Formatter/Linter:** [Ruff](https://docs.astral.sh/ruff/) - line length 100, targeting Python 3.12.
- **Type checking:** [Pyright](https://github.com/microsoft/pyright) in strict mode. All code must be fully typed.
- **Docstrings:** [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
- Pre-commit hooks enforce formatting, trailing whitespace, and valid YAML/TOML automatically on each commit.

## Submitting changes

1. Create a feature branch from `main`:

   ```bash
   git checkout -b my-feature
   ```

2. Make your changes and commit with a clear, descriptive message.
3. Push your branch and open a pull request against `main`.
4. CI runs on all pull requests — ensure all checks are green before requesting review.

## Reporting issues

Open a GitHub issue with:

- A clear description of the problem or feature request
- Steps to reproduce (for bugs)
- Python version and OS
