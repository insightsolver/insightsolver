# InsightSolver

**InsightSolver** is a solution for advanced data insights powered by a centralized cloud-based rule-mining engine.
It enables organizations to uncover hidden patterns, generate actionable insights, and make smarter data-driven decisions.
This repository hosts the Python-based *InsightSolver API client*.

## 🚀 Getting started

To get started, you need the following:

1. The `insightsolver` Python module installed.
2. A service key.
3. Credits to use the API.

## 🛠️ Installation

You can install the `insightsolver` Python module in different ways:

1. *100% CLI*. If you have git installed and you don't need a local copy of the repo: 
```bash
pip install git+https://github.com/insightsolver/insightsolver.git
```
2. *100% CLI*. If you have git installed and you want also a local copy of the repo:
```bash
git clone https://github.com/insightsolver/insightsolver.git
cd insightsolver
pip install .
```
3. *50% GUI + 50% CLI*. If you don't have git installed, in a browser go to [https://github.com/insightsolver/insightsolver](https://github.com/insightsolver/insightsolver) and clic on the green button ```<> Code``` then ```Download Zip```. Then open the zip file, then with a CLI ```cd``` to the unzipped folder then do:
```bash
pip install .
```
4. *100% CLI*. *(coming soon)* From PyPi: `pip install insightsolver`.

Because the current GitHub repo is private, the first two methods need a github account with an active [personal access token (classic)](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

*Warning for Anaconda users:* When using a virtual environment managed by [Anaconda](http://anaconda.org), the installation of the `insightsolver` library as specified above could install dependencies (specified in the file `requirements.txt`) via pip that are not handled by Anaconda. There are two options available. The first option is to do as specified above, which lets pip install dependencies, but risk that the virtual environment is no longer handled by Anaconda. The second option is to add a `--no-deps` flag to the pip install, e.g. `pip install --no-deps .`. This last command would install the scripts of the `insightsolver` module without installing the dependencies. This prevents breaking the Anaconda environment but could result in `insightsolver` not finding all the required dependencies at runtime. These required dependencies should therefore either be installed manually from within the Anaconda application or either using the `environment.yml` file.

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

- `m` is the number of rows,
- `n` is the number of columns,
- `ceil` is the mathematical ceiling function (rounds up to the next integer).

Here are some examples:

| Rows (`m`) | Columns (`n`)  | Computation           | Credits Charged  |
|------------|----------------|-----------------------|------------------|
| 1000       | 10             | ceil(1000*10/10000)   | 1                |
| 10000      | 25             | ceil(10000*25/10000)  | 25               |
| 20000      | 100            | ceil(20000*100/10000) | 200              |

> For reference, the Titanic training dataset from [Kaggle](https://www.kaggle.com/competitions/titanic) has **891 rows** and **10 columns** (excluding `PassengerId`), which results in:
>
> ```python
> ceil(891 * 10 / 10000) = 1 credit
> ```
>
> So you can think of **1 credit as roughly "one Titanic"** in size.

*Tips to reduce credit usage:*

- Remove unused or irrelevant columns,
- Filter the dataset before sending it,
- Work with samples when appropriate.

## 📚 Documentation

Comprehensive technical documentation for the `insightsolver` module is available here:

- [PDF version](https://github.com/insightsolver/insightsolver/blob/main/doc/insightsolver_api_client.pdf).
- ReadTheDocs.com *(coming soon)*.

## 📄 Changelog

Here you'll find the [changelog](./changelog.md).

## 📦 Dependencies

- Python 3.9 or higher
- pandas, numpy, requests, google-auth, cryptography, mpmath, etc..

## ⚖️ License

Here you'll find the [LICENSE](./LICENSE).

### 🗃️ Third-Party Licenses

This project relies on third-party open-source Python packages, used in:

- the **client-side API module** (installable via `pip`),
- the **server-side API backend**, and
- the **web application frontend**.

To ensure transparency and fulfill licensing obligations, we provide a full list of dependencies along with their license information in the file [`THIRD_PARTY_LICENSES.csv`](./THIRD_PARTY_LICENSES.csv).  
It includes:
- the package name and version,
- the license type, and
- a link to the package’s source or home page.

All third-party libraries are used unmodified and installed from [PyPI](https://pypi.org).

## 🤝 Contact

- Email [support@insightsolver.com](mailto:support@insightsolver.com)
- Official website: [www.insightsolver.com](https://www.insightsolver.com)
- [LinkedIn](https://www.linkedin.com/company/insightsolver/)



