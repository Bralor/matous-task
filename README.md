# Clusterer
A command-line script for clustering the data.

<br>

## Installation
It's not mandatory but always work with *virtual environment*
and *virtual interpreter*:
```bash
# Install pyenv and choose the version of the interpreter
pyenv local <python-version>  # pyenv local 3.9.9
python -m venv env
```

Ensure `build`, `setuptools`, and `wheel` are installed and up-to-date:
```bash
python -m pip install --upgrade build setuptools wheel
```

Build the package:
```bash
python -m build
```

Install the package:
```bash
python -m pip install dist/matous_clusterer-*.whl
```

<br>

## Usage

<br>

## Options

<br>

## Tests
After the installation process you can run all the testing suites:
```bash
python -m pytest -v src/tests
```