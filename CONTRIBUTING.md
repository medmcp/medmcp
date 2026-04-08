# Contributing to MedMCP

Thank you for your interest in contributing to MedMCP!
This guide will help you get set up and familiar with our development workflow.

## Get started!

Ready to contribute? Here's how to set up `medmcp` for local development.

### 1) Create an issue on the GitHub repository

It's good practice to first discuss the proposed changes as the feature might
already be implemented.


### 2) Fork the `medmcp` repository on GitHub

Click [here](https://github.com/medmcp/medmcp-dev/fork) to create your fork.

### 3) Clone your fork locally

```bash
git clone https://github.com/<your-username>/medmcp-dev.git
cd medmcp-dev
```

### 4) Install your local copy into a virtual environment

[uv](https://docs.astral.sh/uv/) is recommended for development. You can simply install it with:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

We provide simple [just](https://just.systems/) commands to set up the rest of the development environment. Just can be installed with:
```bash
uv tool install rust-just
```

You can now install `medmcp` and all its dependencies with:
```bash
just setup
```
The command (1) checks if [uv](https://docs.astral.sh/uv/) is available if necessary installs it;
(2) checks if [Ollama](https://ollama.com/) is available and if necessary installs it;
(3) installs all other dependencies.


### 5) Running checks locally

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
