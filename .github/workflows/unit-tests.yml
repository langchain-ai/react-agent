# This workflow will run unit tests for the current project

name: CI

on:
  push:
    branches: ["main"]
  pull_request:
  workflow_dispatch: # Allows triggering the workflow manually in GitHub UI

permissions:
  contents: read

# If another push to the same PR or branch happens while this workflow is still running,
# cancel the earlier run in favor of the next run.
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  unit-tests:
    name: Unit Tests
    strategy:
      matrix:
        os: [ubuntu-latest]
        python-version: ["3.11", "3.12"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          curl -LsSf https://astral.sh/uv/install.sh | sh
          uv venv
          uv pip install -r pyproject.toml
      - name: Lint with ruff
        run: |
          uv pip install ruff
          uv run ruff check .
      - name: Lint with mypy
        run: |
          uv pip install mypy
          uv run mypy --strict src/
      - name: Check README spelling
        uses: codespell-project/actions-codespell@406322ec52dd7b488e48c1c4b82e2a8b3a1bf630 # v2
        with:
          ignore_words_file: .codespellignore
          path: README.md
      - name: Check code spelling
        uses: codespell-project/actions-codespell@406322ec52dd7b488e48c1c4b82e2a8b3a1bf630 # v2
        with:
          ignore_words_file: .codespellignore
          path: src/
      - name: Run tests with pytest
        run: |
          uv pip install pytest
          uv run pytest tests/unit_tests
