# insightsolver
This is the repo for the *InsightSolver API client*.  InsightSolver is a solution for advanced data insights powered by a centralized rule-mining engine. It helps organizations uncover hidden patterns, generate actionable insights, and make data-driven decisions.

## Installation

You can install the InsightSolver API client in different ways:

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

## Quick start

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

## Documentation

You can find a pdf version of the technical documentation [here](https://github.com/insightsolver/insightsolver/doc/insightsolver_api_client.pdf).

## Dependencies

- Python 3.9 or higher
- pandas, numpy, requests, jsonpickle, google-auth, cryptography, mpmath.

## License

See the [LICENSE](./LICENSE) file for more details.

## Contact

[support@insightsolver.com](mailto:support@insightsolver.com)

