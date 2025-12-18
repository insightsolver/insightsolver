# InsightSolver

[![PyPI version](https://badge.fury.io/py/insightsolver.svg)](https://badge.fury.io/py/insightsolver)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://github.com/insightsolver/insightsolver/actions/workflows/ci.yml/badge.svg)](https://github.com/insightsolver/insightsolver/actions)

**InsightSolver** is a solution for advanced data insights powered by a centralized cloud-based rule-mining engine.
It enables organizations to uncover hidden patterns, generate actionable insights, and make smarter data-driven decisions.
This repository hosts the Python-based *InsightSolver API client*.

## 🚀 Getting started

To get started, you need the following:

1. The `insightsolver` Python module installed.
2. A service key.
3. Credits to use the API.

## 🛠️ Installation

You can install the `insightsolver` Python module directly from PyPI:

```bash
pip install insightsolver
```

Or for the latest development version from GitHub:

```bash
pip install git+https://github.com/insightsolver/insightsolver.git
```

## ⚡ Quick start

```python
# Import data
import pandas as pd
df = pd.read_csv('kaggle_titanic_train.csv',index_col='PassengerId')
# Declare a solver
from insightsolver import InsightSolver
solver = InsightSolver(
	df          = df,
	target_name = 'Survived',
	target_goal = 1,
)
# Fit the solver
solver.fit(
	service_key = 'your_service_key.json',
)
# Print the result
solver.print()
# Plot the result
solver.plot()
```
A demo can also be found in [here](https://github.com/insightsolver/insightsolver/blob/main/demo/demo_insightsolver.py)

## 💳 Credit Consumption

The API charges usage based on the **size of the dataset** you submit.
The number of credits is calculated as:

```python
credits = ceil(m * n / 10000)
```

where:

- `m` is the number of rows (excluding the header),
- `n` is the number of feature columns (excluding the index column, the target column and other ignored columns),
- `ceil` is the mathematical ceiling function (rounds up to the next integer).

Here are some examples:

| Rows (`m`) | Columns (`n`)  | Computation           | Credits Charged  |
|------------|----------------|-----------------------|------------------|
| 1000       | 10             | ceil(1000*10/10000)   | 1                |
| 10000      | 25             | ceil(10000*25/10000)  | 25               |
| 20000      | 100            | ceil(20000*100/10000) | 200              |

> For reference, the Titanic training dataset from [Kaggle](https://www.kaggle.com/competitions/titanic) has **m=891 rows** and **n=9 feature columns** (excluding `PassengerId` and `Survived`), which results in:
>
> ```python
> ceil(891 * 9 / 10000) = 1 credit
> ```
>
> So you can think of **1 credit as roughly "one Titanic"** in size.

*Tips to reduce credit usage:*

- Remove unused or irrelevant columns or set them to `'ignore'`,
- Filter the rows of the dataset,
- Samples the rows of the dataset.

## 📚 Documentation

Comprehensive technical documentation for the `insightsolver` module is available here:

- [PDF version](https://github.com/insightsolver/insightsolver/blob/main/doc/insightsolver_api_client.pdf).
- [Sphinx version](https://insightsolver.github.io/sphinx/index.html)

## 📄 Changelog

Here you'll find the [changelog](./changelog.md).

## 📦 Dependencies

- Python 3.9 or higher
- pandas, numpy, requests, google-auth, cryptography, mpmath, etc..

## ⚖️ License

The **InsightSolver API client** library is licensed under the **Apache License 2.0**:
- You can use, modify, and redistribute it freely in your projects, including commercial ones.
- This software is provided ‘as-is’, without warranty of any kind, express or implied, including but not limited to merchantability or fitness for a particular purpose.

See the full [LICENSE](./LICENSE) file for details.

**Note:** The **InsightSolver API server** is proprietary and requires a valid subscription to use. The **InsightSolver API client** library provides a **client** interface only; usage of the **server** is subject to our terms of service.

### 🗃️ Third-Party Licenses

The **client-side API module** (installable via `pip`) uses third-party open-source Python packages.

To ensure transparency and comply with licensing requirements, we provide a complete list of these dependencies in [`THIRD_PARTY_LICENSES.csv`](./THIRD_PARTY_LICENSES.csv). The file includes:

- Package name and version  
- License type  
- Link to the package’s source or homepage  

All third-party libraries are used **unmodified** and installed directly from [PyPI](https://pypi.org).

This information is provided to help users and organizations verify compliance with open-source licenses when integrating the client library into their projects.

## 🤝 Contact

- Email [support@insightsolver.com](mailto:support@insightsolver.com)
- Official website: [insightsolver.com](https://www.insightsolver.com)
- GitHub website: [insightsolver.github.io](https://insightsolver.github.io)
- [LinkedIn](https://www.linkedin.com/company/insightsolver/)



