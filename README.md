# InsightSolver

**InsightSolver** is a solution for advanced data insights powered by a centralized cloud-based rule-mining engine.
It enables organizations to uncover hidden patterns, generate actionable insights, and make smarter data-driven decisions.
This repository hosts the Python-based *InsightSolver API client*.

## 🚀 Getting started

To get started, you need the following:

1. A service key.
2. The `insightsolver` Python module installed.

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
```
A demo can also be found in [here](https://github.com/insightsolver/insightsolver/blob/main/demo/demo_insightsolver.py)

## 📚 Documentation

Comprehensive technical documentation for the `insightsolver` module is available here:

- [PDF version](https://github.com/insightsolver/insightsolver/blob/main/doc/insightsolver_api_client.pdf).
- ReadTheDocs.com *(coming soon)*.

## 📦 Dependencies

- Python 3.9 or higher
- pandas, numpy, requests, jsonpickle, google-auth, cryptography, mpmath.

## ⚖️ License

Here you'll find the [LICENSE](./LICENSE).

## 🤝 Contact

- Email: [support@insightsolver.com](mailto:support@insightsolver.com)
- Official website *(coming soon)*: [www.insightsolver.com](https://www.insightsolver.com).
- LinkedIn: *(coming soon)*



