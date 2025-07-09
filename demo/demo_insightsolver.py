"""
* `Organization`:  InsightSolver Solutions Inc.
* `Project Name`:  InsightSolver
* `Module Name`:   demo_insightsolver
* `Author`:        Noé Aubin-Cadot
* `Email`:         noe.aubin-cadot@datascienceinstitute.ca
* `Last Updated`:  2025-06-20
* `First Created`: 2024-09-09

Description
-----------
This module contains a demo to use the ``InsightSolver`` class.

Note
----
A service key is necessary to use the API client.

License
-------
Exclusive Use License - see `LICENSE <license.html>`_ for details.

----------------------------

"""

################################################################################
################################################################################
# Import some libraries

import pandas as pd

################################################################################
################################################################################
# Defining demo functions

def demo_InsightSolver_Titanic():
	"""
	In this demo we:

	1. Import a demo dataset (here `Kaggle Titanic Train <https://www.kaggle.com/competitions/titanic/data>`_).
	2. Set the solver parameters.
	3. Create an instance of the class ``InsightSolver``.
	4. Specify the service key.
	5. Fit the solver.
	6. Show the results.
	"""

	# 1. Import a demo dataset (Kaggle Titanic Train).
	print("1. Importing data...")
	df = pd.read_csv('kaggle_titanic_train.csv',index_col='PassengerId')

	# 2. Set the solver parameters.
	print("2. Set the solver parameters...")
	# Define the target variable
	target_name = 'Survived' # We are interested in if a passenger survives or not
	# Define the target goal
	target_goal = 1 # We are looking for survivors
	# Specify the types of some variables (optional)
	columns_types = {
		'Name'   : 'ignore',       # Let's ignore the variable Name
		'Ticket' : 'ignore',       # Let's ignore the variable Ticket
		'Cabin'  : 'ignore',       # Let's ignore the variable Cabin
		'Pclass' : 'continuous',   # Let's consider the Pclass as a continuous variable (i.e. ordered)
		'SibSp'  : 'continuous',   # Let's consider SibSp as a continuous variable (i.e. ordered)
		'Parch'  : 'continuous',   # Let's consider Parch as a continuous variable (i.e. ordered)
	}

	# 3. Instantiate a InsightSolver.
	print("3. Create an instance of the class InsightSolver...")
	from insightsolver import InsightSolver
	solver = InsightSolver(
		df            = df,
		target_name   = target_name,
		target_goal   = target_goal,
		columns_types = columns_types,
	)

	# 4. Specify the service key
	print("4. Specify the service key...")
	service_key = 'name_of_your_service_key.json' # Specify the name of your service key file here

	# 5. Fit the solver.
	print("5. Fit the solver...")
	solver.fit(
		service_key = service_key,
	)

	# 6. Show the results.
	print("6. Printing the results...")
	solver.print()

################################################################################
################################################################################
# Executing the demo functions

def main():

	do_demo_InsightSolver_Titanic=1
	if do_demo_InsightSolver_Titanic:
		demo_InsightSolver_Titanic()

if __name__ == '__main__':
	main()


