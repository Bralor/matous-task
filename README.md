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

First things first, create a **simple data sample** (`.json` or `.npy`):
```bash
# sample.npy
[[1, 2], [3, 4]]
```

Then you need to create a **configuration file**, like example with the sample:
```bash
# config.yaml
algorithm: kmeans

input:
  format: numpy
  filepath: .sample.npy       # The file you've already created

output:
  format: numpy
  filepath: .test_output.npy  # The file with results

hyperparameters:
    n_clusters: 3
    random_state: 42
    max_iter: 500
```

Run the entrypoint script with the filepath
to your configuration file `config.yaml`:
```bash
$ python src/main.py config.yaml
```

Currently, the storing procedure is not complete. You can at least read
the output:
```bash
[0 2 0 1 1 1]
```

<br>

## Tests
After the installation process you can run all the testing suites:
```bash
python -m pytest -v 
```