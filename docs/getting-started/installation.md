# Installation

## Prerequisites

| Requirement | Version |
|---|---|
| Python | >= 3.10 (3.12 recommended) |
| [uv](https://docs.astral.sh/uv/) | latest |

## Install with uv

```bash
# Clone the repository
git clone https://github.com/lowmason/alt_nfp.git
cd alt_nfp

# Install all dependencies (including dev tools)
uv sync
```

This installs the package in editable mode along with all production and
development dependencies.

## Optional: Apple Silicon Acceleration

On Apple Silicon Macs, `nutpie` provides a significantly faster NUTS sampler.
It is included in the default dependencies and will be preferred automatically
when available.

## Development Dependencies

The `dev` dependency group includes:

| Package | Purpose |
|---|---|
| `pytest` + `pytest-cov` | Testing and coverage |
| `black` | Code formatting |
| `ruff` | Linting |
| `mypy` | Static type checking |
| `ipython` + `jupyter` | Interactive exploration |

Install dev dependencies:

```bash
uv sync --group dev
```

## Documentation Dependencies

To build the documentation locally:

```bash
uv sync --group docs
```

Then serve:

```bash
uv run mkdocs serve
```

## Verify Installation

```bash
# Run the test suite
uv run pytest tests/ -m "not network"

# Check imports
uv run python -c "import alt_nfp; print(alt_nfp.__doc__[:60])"
```
