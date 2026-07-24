# mad-2026-spring-ml-algorithms

## Getting started (uv)

```bash
# 1) Clone and enter the repo
git clone <repo-url>
cd mad-2026-spring-ml-algorithms

# 2) Create/update .venv and install dependencies from pyproject.toml/uv.lock
uv sync

# 3) (Optional) activate the virtual environment for manual python usage
source .venv/bin/activate
```

## Running the project scripts

```bash
# Show available script targets
uv run python main.py list

# Run diabetes classification analysis
uv run python main.py diabetes

# Run cherry tree linear regression analysis
uv run python main.py cherry-tree

# Run bonus calculation script
uv run python main.py bonuses
```

## Running scripts directly (without main.py dispatcher)

```bash
# Diabetes analysis script
uv run python classification/analyze_diabetes.py

# Cherry tree analysis script
uv run python linear_regression/analyze_cherry_tree.py

# Bonus script
uv run python linear_regression/pay_bonuses_non_modular.py
```