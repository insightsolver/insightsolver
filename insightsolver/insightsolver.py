"""
* `Project Name`:  InsightSolver
* `Module Name`:   insightsolver
* `Author`:        Noé Aubin-Cadot
* `Organization`:  InsightSolver
* `Email`:         noe.aubin-cadot@insightsolver.com
* `Last Updated`:  2025-02-21
* `First Created`: 2024-09-09

Description
-----------
This module contains the ``InsightSolver`` class.
It is meant to make rule mining API calls.

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
# Define some global variables

API_SOURCE_PUBLIC = '-5547756002485302797'
# This string can be shared to customers of DSI as it refers to the public rule mining API.
# It is not tied with a specific user but is only a code that means 'public user outside DSI'.

################################################################################
################################################################################
# Import some libraries

import pandas as pd
import numpy as np
from requests.models import Response
from typing import Optional, Union, Dict, Sequence
import numbers
import mpmath # Useful for when the p-values are very small

################################################################################
################################################################################
# Printing settings

#pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 12)
pd.set_option('max_colwidth', 20)
pd.set_option('display.width', 1000)

# Since numpy 2.0.0 printing numbers show their type but it makes reading results quickly harder.
# We revert to the legacy way of printing numbers.
if np.__version__>='2.0.0':
	legacy = '1.25'
else:
	legacy = None
np.set_printoptions(
	#linewidth = np.inf,
	#precision = 20,
	#precision = 1,
	#threshold = sys.maxsize,
	legacy = legacy,
)

################################################################################
################################################################################
# Defining some utilities

def validate_class_integrity(
	verbose: bool,
	df: Optional[pd.DataFrame] ,
	target_name: Optional[Union[str,int]],
	target_goal: Optional[Union[str,numbers.Real,np.uint8]],
	columns_types: Optional[Dict],
	columns_descr:Optional[Dict],
	threshold_M_max: Optional[int],
	specified_constraints: Optional[Dict],
	top_N_rules: Optional[int],
	filtering_score: Optional[str],
)->None:
	"""
	This function aims to validate the integrity of the InsightSolver class.
	"""
	if verbose:
		print("Validating the integrity of the class...")
	# Validate that target_name is a column of df
	if target_name not in df.columns:
		raise Exception(f"ERROR (target_name invalid): target_name='{target_name}' not in df.columns.")
	# Validate that the columns types are valid
	for column_name in columns_types.keys():
		if column_name not in df.columns:
			raise Exception(f"ERROR (columns_types invalid): the dict 'columns_types' contains the column_name='{column_name}' but it is not a column of df.")
		if columns_types[column_name] not in ['binary','multiclass','continuous','ignore']:
			raise Exception(f"ERROR (columns_types invalid): the column='{column_name}' cannot be of type='{columns_types[column_name]}' because it must be in ['binary','multiclass','continuous','ignore'].")
	# Validate that the target is not ignored
	if target_name in columns_types.keys():
		# Take the target type
		target_type = columns_types[target_name]
		# If the target type is 'ignore' there's a problem
		if target_type=='ignore':
			raise Exception(f"ERROR: target_name='{target_name}' is specified as 'ignore'.")
	# Validate that not all features are ignored
	features_types = columns_types.copy()
	if target_name in features_types.keys():
		features_types.pop(target_name)
	M,n=df.shape
	if (len(features_types)==n-1)&(all(features_types[column_name]=='ignore' for column_name in features_types.keys())):
		raise Exception("ERROR (columns_types): The specified type of each feature is 'ignore'.")
	# Validate that the filtering_score is valid
	if filtering_score==None:
		raise Exception("ERROR (filtering_score): The filtering score cannot be None.")
	elif filtering_score!='auto':
		valid_scores = ['lift','coverage','p_value','F_score','Z_score','TPR','PPV']
		scores = filtering_score.split('&')
		invalid_scores = sorted(set(scores)-set(valid_scores))
		if len(invalid_scores)>0:
			raise Exception(f"ERROR (filtering_score): The filtering score is not valid because it contains these scores: {invalid_scores}.")

def format_value(
	value,
	format_type = 'scientific', # 'scientific', 'percentage', 'scientific_no_decimals'
	decimals    = 1,
	verbose     = False,
):
	"""
	This function formats values depending on the type of values (float or mpmath) and the type of the format to show:
	- 'percentage' : shows the values as percentage (default)
	- 'scientific' : shows the values in scientific notation with 4 decimals
	- 'scientific_no_decimals' : shows the values in scientific notation without decimals
	"""
	if pd.isna(value):
		return ''

	if isinstance(value, mpmath.mpf):
		if value>1e-320:
			if verbose:
				print(f"The value mpmath {value} > 1e-320 is converted to float, no need to keep mpmath.")
			value = float(value)
	
	if format_type == 'percentage':
		if isinstance(value, mpmath.mpf):
			return f"{mpmath.nstr(value * 100, n=1+decimals, strip_zeros=True)}%"
		elif isinstance(value, float):
			return f"{value * 100:.{decimals}f}%"    
	elif format_type == 'scientific':
		if isinstance(value, mpmath.mpf):
			return mpmath.nstr(value, n=1+decimals, strip_zeros=False)
		elif isinstance(value, float):
			return f'{value:.{decimals}e}'
	elif format_type == 'scientific_no_decimals':
		if isinstance(value, mpmath.mpf):
			result = mpmath.nstr(value, n=1, strip_zeros=False)
			if '.e' in result:
				result = result.replace('.e', 'e')
			return result
		elif isinstance(value, float):
			return f"{value:.0e}"
	else:
		return value

def S_to_index_points_in_rule(
	solver,
	S:dict,
	verbose:bool              = False,
	df:Optional[pd.DataFrame] = None,
)->pd.Index:
	"""
	This function takes a rule S and returns the index of the points inside the rule of a DataFrame.
	If no DataFrame is provided, the one used to train the solver is used.
	"""
	if verbose:
		print('S :',S)
	# Create a temporary DataFrame that will be iteratively filtered
	if isinstance(df,pd.DataFrame):
		# If df is specified, we take it
		df_features_filtre = df.copy()
	else:
		# If df is not specified, we take the DataFrame in the solver
		df_features_filtre = solver.df.copy()
	# Take the features names in the rule S
	feature_names = list(S.keys())
	# Make sure that the names are legit
	feature_names_illegal = set(feature_names)-set(solver.df.columns)
	if len(feature_names_illegal)>0:
		raise Exception(f"ERROR: there are illegal names in the rule S : {feature_names_illegal}.")
	# Sort the features names from the rule S
	feature_names.sort()
	if verbose:
		print('feature_names :',feature_names)
	# Loop over the features of the rule S
	for feature_name in feature_names:
		if verbose:
			print('• feature_name :',feature_name)
		# Take the value of the feature in the rule S
		feature_S = S[feature_name]
		if verbose:
			print('•- feature_S :',feature_S)
		# Take the btype of the feature
		feature_type = solver.columns_types[feature_name]
		if verbose:
			print('•- feature_type :',feature_type)
		"""
		The types :
		- binary
		- multiclass
		- continuous
		- ignore
		"""
		# Depending on the type of the feature the data is filtered differently
		if feature_type=='ignore':
			# If the variable has the type 'ignore', we skip it.
			continue
		elif feature_type in ['binary','multiclass']:
			# If the variable is a categorical variable
			if feature_S==set():
				# If the variable can take no modality, return an empty list
				return pd.Index([],name=df_features_filtre.index.name)
			if isinstance(feature_S,int):
				# If the rule is an integer, convert to a set with one element
				feature_S = {feature_S}
			elif isinstance(feature_S,float):
				# If the rule is a float, convert to a set with one element
				if pd.isna(feature_S):
					# If the float is a NaN, convert it to 'nan'
					feature_S = 'nan'
				elif int(feature_S)==feature_S:
					# If the float is an integer, convert it to an integer (to have 0 or 1 instead of 0.0 or 1.0)
					feature_S = int(feature_S)
				# Convert to a set with one element
				feature_S = {feature_S}
			elif isinstance(feature_S,str):
				# If the rule is a string, convert to a set with one element
				feature_S = {feature_S}
			if verbose:
				print('•- feature_S :',feature_S)
			# Create a mask to filter the DataFrame
			mask = pd.Series(
				data  = False,
				index = df_features_filtre.index,
			)
			# Loop over the modalities of the rule S
			for modality in feature_S:
				if modality in [np.nan,'nan']:
					# If the modality is NaN
					# Keep the NaNs
					mask = mask|df_features_filtre[feature_name].isna()
				elif modality=='other':
					# If the modality is 'other'
					if 'other' in df_features_filtre[feature_name]:
						# If 'other' is a modality of the original data
						# Keep the modality 'other'
						mask = mask|(df_features_filtre[feature_name]=='other')
					if feature_name in solver.other_modalities.keys():
						# If the feature is present in the conversion to other modalities
						if len(solver.other_modalities[feature_name])>0:
							# If at least one modality was mapped to 'other'
							# Take the other modalities
							other_modalities = solver.other_modalities[feature_name]
							# Keep the other modalities
							mask = mask|(df_features_filtre[feature_name].isin(other_modalities))
				elif modality in df_features_filtre.values:
					# If the modality is in the original data
					# Keep the modality
					mask = mask|(df_features_filtre[feature_name]==modality)
				elif str(modality) in df_features_filtre.values:
					# If str(modality) is in the original data
					# Keep str(modality)
					print("WARNING: 'str(modality)' is in the data but not 'modality'.")
					mask = mask|(df_features_filtre[feature_name]==str(modality))
				else:
					raise Exception(f"ERROR: the modality='{modality}' is not in the data.")
			# Filter the DataFrame by the mask
			df_features_filtre = df_features_filtre[mask]
		elif feature_type=='continuous':
			# If the feature is continuous
			if feature_S[1] in ['exclude_nan','include_nan']:
				# If the feature is continuous with NaNs
				[[s_rule_min,s_rule_max],include_or_exlude_nan] = feature_S
				if verbose:
					print('•- s_rule_min =',s_rule_min)
					print('•- s_rule_max =',s_rule_max)
					print('•- include_or_exlude_nan =',include_or_exlude_nan)
				# Take the continuous values of the feature
				s = df_features_filtre[feature_name]
				# Keep only the values between the interval
				mask = (s_rule_min<=s)&(s<=s_rule_max)
				# Handle NaNs
				if include_or_exlude_nan=='exclude_nan':
					# NaNs are a priori excluded because (s_rule_min<=s)&(s<=s_rule_max) can only be True for non NaNs.
					...
				elif include_or_exlude_nan=='include_nan':
					# If we want to include NaNs
					mask = mask|s.isna()
				else:
					raise Exception(f"ERROR: include_or_exlude_nan='{include_or_exlude_nan}' should be either 'include_nan' or 'exclude_nan'.")
				# Filter the data
				df_features_filtre = df_features_filtre[mask]
			else:
				# If the feature is continuous without NaNs
				s_rule_min,s_rule_max = feature_S
				if verbose:
					print('•- s_rule_min =',s_rule_min)
					print('•- s_rule_max =',s_rule_max)
				# Take the continuous values of the feature
				s = df_features_filtre[feature_name]
				# Keep only the values between the interval
				mask = (s_rule_min<=s)&(s<=s_rule_max)
				# Filter the data
				df_features_filtre = df_features_filtre[mask]
	# Take the index
	index = df_features_filtre.index
	# Sort the index
	index = index.sort_values()
	# Return the index
	return index

################################################################################
################################################################################
# Defining the solver class

class InsightSolver:
	"""
	The class ``InsightSolver`` is meant to :

	1. Take input data.
	2. Make an insightsolver API calls to the server.
	3. Present the results of the rule mining.

	Attributes
	----------
	df: DataFrame
		The DataFrame that contains the data to analyse.
	target_name: str (default None)
		Name of the target variable (by default it's the first column).
	target_goal: (str or int)
		Target goal.
	target_threshold: (int or float)
		Threshold used to convert a continuous target variable to a binary target variable.
	M: int
		Number of points in the population.
	M0: int
		Number of points 0 in the population.
	M1: int
		Number of points 1 in the population.
	columns_types: dict
		Types of the columns.
	columns_descr: dict
		Textual descriptions of the columns.
	other_modalities: dict
		Modalities that are mapped to the modality 'other'.
	threshold_M_max: int (default 10000)
		Threshold on the maximum number of observations to consider, above which we under sample the observations to 10000.
	specified_constraints: dict
		Dictionary of the specified constraints on ``m_min``, ``m_max``, ``coverage_min``, ``coverage_max``.
	top_N_rules: int (default 10)
		An integer that specifies the maximum number of rules to get from the rule mining.
	filtering_score: str (default 'auto')
		A string that specifies the filtering score to be used when selecting rules.
	n_benchmark_original: int (default 5)
		An integer that specifies the number of benchmarking runs to execute where the target is not shuffled.
	n_benchmark_shuffle: int (default 20)
		An integer that specifies the number of benchmarking runs to execute where the target is shuffled.
	monitoring_metadata: dict
		Dictionary of monitoring metadata.
	benchmark_scores: dict
		Dictionary of the benchmarking scores against shuffled data.
	rule_mining_results: dict
		Dictionary that contains the results of the rule mining.

	Methods
	-------
	validate_class_integrity: None
		Validates the integrity of the class.
	ingest_dict: None
		Ingests a Python dict.
	ingest_json_string: None
		Ingests a JSON string.
	fit: None
		Fits the solver.
	S_to_index_points_in_rule: Pandas Index
		Returns the index of the points in a rule S.
	S_to_s_points_in_rule: Pandas Series
		Returns a boolean Pandas Series that tells if the point is in the rule S.
	S_to_df_filtered: Pandas DataFrame
		Returns the filtered df of rows that are in the rule S.
	ruleset_count: int
		Counts the number of rules held by the InsightSolver.
	i_to_rule: dict
		Gives the rule i of the InsightSolver.
	i_to_subrules_dataframe: Pandas DataFrame
		Returns a DataFrame containing the informations about the subrules of the rule i.
	i_to_feature_contributions_S: Pandas DataFrame
		Returns a DataFrame of the feature contributions of the variables in the rule S at position i.
	i_to_print: None
		Prints the content of the rule i in the InsightSolver.
	get_range_i: list
		Gives the range of i in the InsightSolver.
	print: None
		Prints the content of the InsightSolver.
	print_light: None
		Prints the content of the InsightSolver ('light' mode).
	print_dense: None
		Prints the content of the InsightSolver ('dense' mode).
	to_dict: dict
		Exports the content of the InsightSolver object to a Python dict.
	to_json_string: str
		Exports the content of the InsightSolver object to a JSON string.
	to_dataframe: Pandas DataFrame
		Exports the rule mining results to a Pandas DataFrame.
	to_csv: str
		Exports the rule mining results to a CSV string and/or a local CSV file.

	Example
	-------
	Here's a sample code to use the class ``InsightSolver``::

		# Specify the service key
		service_key = 'name_of_your_service_key.json'
		
		# Import some data
		import pandas as pd
		df = pd.read_csv('kaggle_titanic_train.csv')
		
		# Specify the name of the target variable
		target_name = 'Survived' # We are interested in whether the passengers survived or not
		
		# Specify the target goal
		target_goal = 1 # We are searching rules that describe survivors
		
		# Import the class InsightSolver from the module insightsolver
		from insightsolver import InsightSolver
		
		# Create an instance of the class InsightSolver
		solver = InsightSolver(
			df          = df,          # A dataset
			target_name = target_name, # Name of the target variable
			target_goal = target_goal, # Target goal
		)
		
		# Fit the solver
		solver.fit(
			service_key = service_key, # Use your API service key here
		)
		
		# Print the rule mining results
		solver.print()
	"""

	def __init__(
		self,
		verbose: bool                                           = False,  # Verbosity during the initialization of the solver
		df: Optional[pd.DataFrame]                              = None,   # DataFrame in which we want to analyse the data
		target_name: Optional[Union[str,int]]                   = None,   # Name of the target variable
		target_goal: Optional[Union[str,numbers.Real,np.uint8]] = None,   # Target goal
		columns_types: Optional[Dict]                           = dict(), # Types of the columns
		columns_descr:Optional[Dict]                            = dict(), # Descriptions of the columns
		threshold_M_max: Optional[int]                          = 10000,  # Maximum number of observations to consider
		specified_constraints: Optional[Dict]                   = dict(), # Specified constraints on the rule mining
		top_N_rules: Optional[int]                              = 10,     # Maximum number of rules to get from the rule mining
		filtering_score: Optional[str]                          = 'auto', # Filtering score to be used when selecting rules.
		n_benchmark_original: Optional[int]                     = 5,      # Number of benchmarking runs to execute without shuffling.
		n_benchmark_shuffle: Optional[int]                      = 20,     # Number of benchmarking runs to execute with shuffling.
	):
		"""
		The initialization occurs when an ``InsightSolver`` class instance is created.

		Parameters
		----------
		verbose: bool (default False)
			If we want the initialization to be verbose.
		df: DataFrame
			The DataFrame that contains the data to analyse (a target column and various feature columns).
		target_name: str
			Name of the column of the target variable.
		target_goal: str (or other modality of the target variable)
			Target goal.
		columns_types: dict
			Types of the columns.
		columns_descr: dict
			Descriptions of the columns.
		threshold_M_max: int
			Threshold on the maximum number of observations to consider, above which we sample observations.
		specified_constraints: dict
			Dictionary of the specified constraints on m_min, m_max, coverage_min, coverage_max.
		top_N_rules: int (default 10)
			An integer that specifies the maximum number of rules to get from the rule mining.
		filtering_score: str (default 'auto')
			A string that specifies the filtering score to be used when selecting rules.
		n_benchmark_original: int (default 5)
			An integer that specifies the number of benchmarking runs to execute where the target is not shuffled.
		n_benchmark_shuffle: int (default 20)
			An integer that specifies the number of benchmarking runs to execute where the target is shuffled.
		
		Returns
		-------
		solver: InsightSolver
			An instance of the class InsightSolver.

		Example
		-------
		Here's a sample code to instantiante the class ``InsightSolver``::

			# Import the class InsightSolver from the module insightsolver
			from insightsolver import InsightSolver

			# Create an instance of the class InsightSolver
			solver = InsightSolver(
				df          = df,          # A dataset
				target_name = target_name, # Name of the target variable
				target_goal = target_goal, # Target goal
			)
		"""
		if verbose:
			print('Initializing an instance of the class InsightSolver...')
		# Validate the integrity of the class
		validate_class_integrity(
			verbose               = verbose,
			df                    = df,
			target_name           = target_name,
			target_goal           = target_goal,
			columns_types         = columns_types,
			columns_descr         = columns_descr,
			threshold_M_max       = threshold_M_max,
			specified_constraints = specified_constraints,
			top_N_rules           = top_N_rules,
			filtering_score       = filtering_score,
		)
		# Handling threshold_M_max
		if threshold_M_max==None:
			threshold_M_max = 10000
		elif threshold_M_max>10000:
			threshold_M_max = 10000
		self.threshold_M_max = threshold_M_max
		# Sample df
		if len(df)>self.threshold_M_max:
			# Sample df locally to limit the amount of data sent to the server.
			# The server will only accept at most 10000 rows.
			self.df = df.sample(
				n            = self.threshold_M_max,
				random_state = 0,
			)
		else:
			# No need to sample the data
			self.df = df.copy()
		# Execution metadata
		self.target_name             = target_name
		self.target_goal             = target_goal
		self.target_threshold        = None
		self.M                       = None
		self.M0                      = None
		self.M1                      = None
		self.columns_types           = columns_types
		self.columns_descr           = columns_descr
		self.other_modalities        = None
		self.specified_constraints   = specified_constraints
		self.top_N_rules             = top_N_rules
		self.filtering_score         = filtering_score
		self.n_benchmark_original    = n_benchmark_original
		self.n_benchmark_shuffle     = n_benchmark_shuffle
		# Monitoring metadata
		self.monitoring_metadata     = dict()
		# Benchmarking scores
		self.benchmark_scores        = dict()
		# Rule mining results
		self.rule_mining_results     = dict()
	def ingest_dict(
		self,
		d: dict,               # The dict to ingest
		verbose: bool = False, # The verbosity
	)->None:
		"""
		This method aims to ingest a Python dict in the solver.
		"""
		# dataset_metadata
		if verbose:
			print('Reading dataset_metadata...')
		if 'dataset_metadata' in d:
			if 'target_threshold' in d['dataset_metadata']:
				self.target_threshold = d['dataset_metadata']['target_threshold']
			else:
				self.target_threshold = None
			if 'M' in d['dataset_metadata']:
				self.M  = d['dataset_metadata']['M']
			else:
				self.M = None
			if 'M0' in d['dataset_metadata']:
				self.M0 = d['dataset_metadata']['M0']
			else:
				self.M0 = None
			if 'M1' in d['dataset_metadata']:
				self.M1 = d['dataset_metadata']['M1']
			else:
				self.M1 = None
			if 'columns_names_to_descr' in d['dataset_metadata']:
				self.columns_descr = d['dataset_metadata']['columns_names_to_descr']
			else:
				self.columns_descr = None
			if 'features_names_to_other_modalities' in d['dataset_metadata']:
				self.other_modalities = d['dataset_metadata']['features_names_to_other_modalities']
			else:
				self.other_modalities = None
		else:
			print("WARNING : dict does not have key 'dataset_metadata'.")
			self.M                = None
			self.M0               = None
			self.M1               = None
			self.other_modalities = None
		# monitoring_metadata
		if verbose:
			print('Reading monitoring_metadata...')
		if 'monitoring_metadata' in d:
			self.monitoring_metadata = d['monitoring_metadata'].copy()
		else:
			print("WARNING : dict does not have key 'monitoring_metadata'.")
			self.monitoring_metadata = dict()
		# benchmark_scores
		if verbose:
			print('Reading benchmark_scores...')
		if 'benchmark_scores' in d:
			self.benchmark_scores = d['benchmark_scores'].copy()
		else:
			print("WARNING : dict does not have key 'benchmark_scores'.")
			self.benchmark_scores = dict()
		# rule_mining_results
		if verbose:
			print('Reading rule_mining_results...')
		if 'rule_mining_results' in d:
			self.rule_mining_results = d['rule_mining_results']
		else:
			print("WARNING : dict does not have key 'rule_mining_results'.")
			self.rule_mining_results = dict()
	def ingest_json_string(
		self,
		json_string: str,      # JSON string to ingest (encoded using jsonpickle)
		verbose: bool = False, # Verbosity
	)->None:
		"""
		This method aims to ingest a JSON string in the solver.
		"""
		# Convert the json_string to a dict
		import jsonpickle
		d = jsonpickle.decode(json_string)
		self.ingest_dict(d)
		# The keys of the rules are given by jsonpickle as string, we need to convert them to integers
		self.rule_mining_results = {int(k):self.rule_mining_results[k] for k in self.rule_mining_results.keys()}
	def fit(
		self,
		verbose:bool              = False,  # Verbosity
		computing_source:str      = 'auto', # Where to compute the rule mining
		service_key:Optional[str] = None,   # Path+name of the service key
		user_email:Optional[str]  = None,   # User email
		api_source:str            = 'auto', # Source of the API call
		do_compress_data:bool     = False,  # If we want to compress the data for the communications with the server
	)->None:
		"""
		This method aims to fit the solver.
		"""
		if verbose:
			print('Fitting the InsightSolver...')
		# Taking the global variables
		if api_source=='auto':
			api_source = API_SOURCE_PUBLIC
		# Make a rule mining API call
		d_in_original = search_best_ruleset_from_API_public(
			df                      = self.df,
			computing_source        = computing_source,
			input_file_service_key  = service_key,
			user_email              = user_email,
			target_name             = self.target_name,
			target_goal             = self.target_goal,
			columns_names_to_btypes = self.columns_types,
			threshold_M_max         = self.threshold_M_max,
			specified_constraints   = self.specified_constraints,
			top_N_rules             = self.top_N_rules,
			n_benchmark_original    = self.n_benchmark_original,
			n_benchmark_shuffle     = self.n_benchmark_shuffle,
			verbose                 = verbose,
			filtering_score         = self.filtering_score,
			api_source              = api_source,
			do_compress_data        = do_compress_data,
		)
		# Ingest the untransformed incoming dict
		self.ingest_dict(
			d = d_in_original,
		)
	def S_to_index_points_in_rule(
		self,
		S:dict,
		verbose:bool              = False,
		df:Optional[pd.DataFrame] = None,
	)->pd.Index:
		"""
		This method returns the index of the points inside a rule S.
		"""
		# Convert the rule S to an index
		index_points_in_rule = S_to_index_points_in_rule(
			solver  = self,
			S       = S,
			verbose = verbose,
			df      = df,
		)
		# Return the index
		return index_points_in_rule
	def S_to_s_points_in_rule(
		self,
		S:dict,
		verbose:bool              = False,
		df:Optional[pd.DataFrame] = None,
	)->pd.Series:
		"""
		This method returns a boolean Series that tells if the points are in the rule S or not.
		"""
		# Take a look at if df is provided
		if not isinstance(df,pd.DataFrame):
			# If df is not provided we take the one in the solver
			df = self.df
		# Make sure that df is a DataFrame
		if not isinstance(df,pd.DataFrame):
			raise Exception(f"ERROR: df must be a DataFrame but not '{type(df)}'.")	
		# Take the index of the points in the rule S
		index_points_in_rule = self.S_to_index_points_in_rule(
			S       = S,
			verbose = verbose,
			df      = df,
		)
		# Create a Pandas Series that tells if the points are in the rule or not
		s_points_in_rule = pd.Series(
			data  = False,
			index = df.index,
			name  = 'in_S',
			dtype = bool,
		)
		s_points_in_rule.loc[index_points_in_rule] = True
		# Return the result
		return s_points_in_rule
	def S_to_df_filtered(
		self,
		S:dict,
		verbose:bool              = False,
		df:Optional[pd.DataFrame] = None,
	):
		"""
		This method returns the DataFrame of rows of df that lie inside a rule S.
		"""
		# Take a look at if df is provided
		if not isinstance(df,pd.DataFrame):
			# If df is not provided we take the one in the solver
			df = self.df
		# Take the index of the points in the rule S
		index_points_in_rule = self.S_to_index_points_in_rule(
			S       = S,
			verbose = verbose,
			df      = df,
		)
		# Create a copy of the DataFrame of the filtered rows
		df_filtered = df.loc[index_points_in_rule].copy()
		# Return the result
		return df_filtered
	def ruleset_count(
		self,
	)->int:
		"""
		This method returns the number of rules held in the InsightSolver.
		"""
		return len(self.rule_mining_results)
	def i_to_rule(
		self,
		i:int, # Key i of the rule
	)->dict:
		rule_i = self.rule_mining_results[i]
		return rule_i
	def i_to_subrules_dataframe(
		self,
		i:int = 0,  # Number of the rule in the InsightSolver
	)->pd.DataFrame:
		"""
		This method returns a DataFrame which contains the informations about the subrules of the rule i.
		"""

		# Take the rule at position i
		rule_i = self.i_to_rule(i=i)

		# Take the subrules
		subrules_S = rule_i['subrules_S']

		# Convert it to a DataFrame
		if len(subrules_S)>0:
			df_subrules_S = pd.DataFrame.from_dict(
				data   = rule_i['subrules_S'],
				orient = 'columns',
			)
		else:
			print('WARNING: The incoming rule is trivial. An empty DataFrame will be returned.')
			cols = [
				'p_value',
				'Z_score',
				'F_score',
				'M',
				'M0',
				'M1',
				'm',
				'm0',
				'm1',
				'coverage',
				'm1/M1',
				'mu_rule',
				'mu_pop',
				'sigma_pop',
				'F1_pop',
				'lift',
				'complexity',
				'subrule_S',
				'var_name',
				'var_rule',
				'p_value_ratio',
				'mc',
				'm0c',
				'm1c',
				'G_bad_class',
				'G_information',
				'G_gini',
			]
			df_subrules_S = pd.DataFrame(columns=cols)

		# Rename some columns
		d_rename = {
			'var_name' : 'variable',
			'var_rule' : 'rule',
			'm1/M1'    : 'TPR', # Sensitivity = coverage of the 1
			'mu_rule'  : 'PPV', # Precision   = purity
		}
		df_subrules_S.rename(columns=d_rename,inplace=True)

		# Parse the shuffling_scores if they are there
		if ('shuffling_scores' in df_subrules_S.columns)&(len(df_subrules_S)>0):
			if 'p_value' in df_subrules_S['shuffling_scores'].iloc[0]:
				df_subrules_S['p_value_cohen_d'] = df_subrules_S['shuffling_scores'].apply(lambda x:x['p_value']['cohen_d'])
				df_subrules_S['p_value_wy_ratio'] = df_subrules_S['shuffling_scores'].apply(lambda x:x['p_value']['wy_ratio'])
			if 'Z_score' in df_subrules_S['shuffling_scores'].iloc[0]:
				df_subrules_S['Z_score_cohen_d'] = df_subrules_S['shuffling_scores'].apply(lambda x:x['Z_score']['cohen_d'])
				df_subrules_S['Z_score_wy_ratio'] = df_subrules_S['shuffling_scores'].apply(lambda x:x['Z_score']['wy_ratio'])
			if 'F_score' in df_subrules_S['shuffling_scores'].iloc[0]:
				df_subrules_S['F_score_cohen_d'] = df_subrules_S['shuffling_scores'].apply(lambda x:x['F_score']['cohen_d'])
				df_subrules_S['F_score_wy_ratio'] = df_subrules_S['shuffling_scores'].apply(lambda x:x['F_score']['wy_ratio'])

		# Move some columns left
		first_cols = [
			'p_value_ratio',
			'variable',
			'rule',
			'complexity',
			'p_value',
			'F_score',
			'Z_score',
			'TPR',
			'PPV',
			'coverage',
			'm',
			'm0',
			'm1',
		]
		df_subrules_S = df_subrules_S[first_cols + [col for col in df_subrules_S.columns if col not in first_cols]]

		# Return the result
		return df_subrules_S
	def i_to_feature_contributions_S(
		self,
		i: int,                            # Key i of the rule
		do_rename_cols: bool       = True, # If we want to rename some columns
		do_ignore_col_rule_S: bool = True, # Of we want to ignore some columns
	)->pd.DataFrame:
		"""
		This method returns a DataFrame of the feature contributions of the variables in the rule S at position i.
		"""
		df_feature_contributions_S = pd.DataFrame.from_dict(
			data   = self.i_to_rule(i)['feature_contributions_S'],
			orient = 'columns',
		)
		df_feature_contributions_S.index.name = 'feature_name'
		if len(df_feature_contributions_S)==0:
			print('WARNING: The incoming rule is trivial. An empty DataFrame will be returned.')
		if do_ignore_col_rule_S:
			df_feature_contributions_S.drop(columns='rule_S',inplace=True)
		if do_rename_cols:
			df_feature_contributions_S.columns = [col.replace('_contribution','') for col in df_feature_contributions_S.columns]
		return df_feature_contributions_S
	def i_to_print(
		self,
		i: int,
		indentation: str                       = '',
		do_print_rule_DataFrame: bool          = False,
		do_print_subrules_S: bool              = True,
		do_show_coverage_diff: bool            = False,
		do_show_cohen_d: bool                  = True,
		do_show_wy_ratio: bool                 = True,
		do_print_feature_contributions_S: bool = True,
	)->None:
		"""
		This method prints the content of the rule i in the InsightSolver.
		"""
		rule_i = self.i_to_rule(i=i)
		print(f'{indentation}p_value         :',rule_i['p_value'])
		print(f'{indentation}F_score         :',rule_i['F_score'])
		print(f'{indentation}Z_score         :',rule_i['Z_score'])
		print(f'{indentation}M               :',self.M)  # peut être redondant
		print(f'{indentation}M0              :',self.M0) # peut être redondant
		print(f'{indentation}M1              :',self.M1) # peut être redondant
		print(f'{indentation}m               :',rule_i['m'])
		print(f'{indentation}m0              :',rule_i['m0'])
		print(f'{indentation}m1              :',rule_i['m1'])
		print(f'{indentation}m/M (coverage)  :',rule_i['coverage'])
		print(f'{indentation}m1/M1    (TPR)  :',rule_i['m1/M1'])
		print(f'{indentation}μ_rule   (PPV)  :',rule_i['mu_rule'])
		print(f'{indentation}μ_pop           :',rule_i['mu_pop'])    # peut être redondant
		print(f'{indentation}σ_pop           :',rule_i['sigma_pop']) # peut être redondant
		print(f'{indentation}lift            :',rule_i['lift'])
		print(f'{indentation}complexity_S    :',rule_i['complexity_S'])
		print(f'{indentation}F1_pop          :',rule_i['F1_pop'])
		if 'G_bad_class' in rule_i.keys():
			print(f'{indentation}G_bad_class     :',rule_i['G_bad_class'])
		if 'G_information' in rule_i.keys():
			print(f'{indentation}G_information   :',rule_i['G_information'])
		if 'G_gini' in rule_i.keys():
			print(f'{indentation}G_gini          :',rule_i['G_gini'])
		rule_S = rule_i['rule_S']
		print(f'{indentation}rule_S          :',rule_S)
		p_value_ratio_S = {k:v for k,v in rule_i['p_value_ratio_S'].items() if k in rule_i['rule_S'].keys()}
		print(f'{indentation}p_value_ratio_S :',p_value_ratio_S)

		if do_print_rule_DataFrame:
			# On calcule le DataFrame de la règle par variables
			df_rules_and_p_value_ratio = pd.concat(
				(
					pd.Series(rule_S).rename('rule'),
					pd.Series(p_value_ratio_S).rename('p_value_ratio'),
				),
				axis=1,
			).reset_index(
				drop=False,
			).rename(
				columns={'index':'variable'},
			).sort_values(
				by='p_value_ratio',
				ascending=True,
			).reset_index(
				drop=True,
			)
			# On ajoute la complexité
			df_rules_and_p_value_ratio['complexity'] = range(1,len(df_rules_and_p_value_ratio)+1)
			# On montre le DataFrame de la règle
			print(f'\nDataFrame of the components and p_value_ratio :')
			print(df_rules_and_p_value_ratio)
		if do_print_subrules_S:
			print('\nDataFrame of the cumulative subrules of S according to the ratio_drop :')
			df_subrules_S = self.i_to_subrules_dataframe(i=i)
			# Select the columns to show
			cols = ['p_value_ratio']
			cols += [
					'variable',
					'rule',
					'complexity',
					'p_value',
					'F_score',
					'Z_score',
					#'G_bad_class',
					'G_information',
					#'G_gini',
					'TPR',
					'PPV',
					'lift',
					'coverage',
					'm',
					'm1',
				]
			if do_show_cohen_d&('Z_score_cohen_d' in df_subrules_S.columns):
				cols += [
					'cohen_d',
				]
				df_subrules_S.rename(columns={'Z_score_cohen_d':'cohen_d'},inplace=True)
			if do_show_wy_ratio&('Z_score_wy_ratio' in df_subrules_S.columns):
				cols += [
					'wy_ratio',
				]
				df_subrules_S.rename(columns={'Z_score_wy_ratio':'wy_ratio'},inplace=True)
			# Si on veut montrer la différence des coverage successifs des sous-règles
			if do_show_coverage_diff:
				df_subrules_S['coverage_diff'] = df_subrules_S['coverage'].diff()
				df_subrules_S.loc[0,'coverage_diff'] = df_subrules_S.loc[0,'coverage']-1
				cols += ['coverage_diff']
			# On ne garde que certaines colonnes
			df_subrules_S = df_subrules_S[cols]
			# On print le DataFrame
			df_subrules_S_formatted = df_subrules_S.copy()
			do_compactify_print=1
			if do_compactify_print:
				df_subrules_S_formatted['p_value_ratio'] = df_subrules_S_formatted['p_value_ratio'].apply(lambda x:format_value(value=x,format_type='scientific',decimals=4))
				df_subrules_S_formatted['p_value']       = df_subrules_S_formatted['p_value'].map(lambda x:format_value(value=x,format_type='scientific',decimals=4))
				df_subrules_S_formatted['F_score']       = df_subrules_S_formatted['F_score'].map('{:.4f}'.format)
				df_subrules_S_formatted['Z_score']       = df_subrules_S_formatted['Z_score'].map('{:.4f}'.format)
				df_subrules_S_formatted['TPR']           = df_subrules_S_formatted['TPR'].map('{:.4f}'.format)
				df_subrules_S_formatted['PPV']           = df_subrules_S_formatted['PPV'].map('{:.4f}'.format)
				df_subrules_S_formatted['lift']          = df_subrules_S_formatted['lift'].map('{:.4f}'.format)
				df_subrules_S_formatted['coverage']      = df_subrules_S_formatted['coverage'].map('{:.4f}'.format)
				if 'cohen_d' in df_subrules_S_formatted.columns:
					df_subrules_S_formatted['cohen_d'] = df_subrules_S_formatted['cohen_d'].map('{:.4f}'.format)
				if 'wy_ratio' in df_subrules_S_formatted.columns:
					df_subrules_S_formatted['wy_ratio'] = df_subrules_S_formatted['wy_ratio'].map('{:.4f}'.format)
			df_subrules_S_str = df_subrules_S_formatted.rename(
				columns = {'complexity':'c'},
			).to_string(
				index = False,
				#float_format = '{:,.4f}'.format
			)
			print(df_subrules_S_str)
		if do_print_feature_contributions_S:
			print('\nDataFrame of the feature contributions of the variables of S :')
			df_feature_contributions_S = self.i_to_feature_contributions_S(
				i              = i,
				do_rename_cols = True,
			)
			print(df_feature_contributions_S)
	def get_range_i(
		self,
		complexity_max: Optional[int] = None,
	)->list:
		"""
		This method gives the range of i in the InsightSolver.
		If the integer complexity_max is specified, return only this number of elements.
		"""
		range_i = sorted(self.rule_mining_results.keys())
		if complexity_max:
			if complexity_max<len(range_i):
				range_i = range_i[:complexity_max]
		return range_i
	def print(
		self,
		verbose: bool                                 = False,  # Verbosity
		r: Optional[int]                              = None,   # Number of rules to print. "None" will print all of them. "1" will print only the first one, "2" will print the 1st and 2nd rule, etc.
		do_print_dataset_metadata: bool               = True,   # If we want to print the dataset metadata.
		do_print_monitoring_metadata: bool            = False,  # If we want to print the monitoring metadata.
		do_print_benchmark_scores:bool                = True,   # If we want to print the benchmarking scores.
		do_show_cohen_d: bool                         = True,   # If we want to print the d of Cohen of the subrules.
		do_show_wy_ratio: bool                        = True,   # If we want to print the WY ratio of the subrules.
		do_print_rule_mining_results: bool            = True,   # If we want to print the rule mining results.
		do_print_rule_DataFrame: bool                 = False,  # If we want to print the the DataFrame of rules.
		do_print_subrules_S: bool                     = True,   # If we want to print the the DataFrame of the subrules of the rules S.
		do_show_coverage_diff: bool                   = False,  # If we want to show the column 'coverage_diff' of the DataFrame of subrules.
		do_print_feature_contributions_S: bool        = True,   # If we want to show the DataFrame of feature importances of the rules S.
		separation_width_between_rules: Optional[int] = 79,     # If we want to show a line between the different rules.
		mode: str                                     = 'full', # The printing mode.
	)->None:
		"""
		This method prints the content of the InsightSolver.
		"""
		if verbose:
			print('Printing the content of the class InsightSolver...')
		if mode not in ['full','light','dense']:
			raise Exception(f"ERROR: mode={mode} must be in ['full','light','dense'].")
		elif mode=='dense':
			# If we want to do a dense print
			self.print_dense()
		elif mode=='light':
			# If we want to do a light print
			self.print_light()
		elif mode=='full':
			if r!=None:
				do_print_dataset_metadata=False
				if r==0:
					r=1 # revert to at least one rule
			if do_print_dataset_metadata:
				# dataset_metadata
				print('\ndataset_metadata :')
				print('target_name      :',self.target_name)
				print('target_goal      :',self.target_goal)
				if self.ruleset_count():
					print('target_threshold :',self.target_threshold)
					print('M                :',self.M)
					print('M0               :',self.M0)
					print('M1               :',self.M1)
			if do_print_monitoring_metadata:
				# monitoring_metadata
				print('\nmonitoring_metadata :')
				if 'p_value_min' in self.monitoring_metadata.keys():
					print('p_value_min        :',self.monitoring_metadata['p_value_min'])
				if 'Z_score_max' in self.monitoring_metadata.keys():
					print('Z_score_max        :',self.monitoring_metadata['Z_score_max'])
				if 'F_score_max' in self.monitoring_metadata.keys():
					print('F_score_max        :',self.monitoring_metadata['F_score_max'])
				if 'precision_p_values' in self.monitoring_metadata.keys():
					print('precision_p_values :',self.monitoring_metadata['precision_p_values'])
			if do_print_benchmark_scores:
				# benchmark_scores
				print('\nbenchmark_scores :')
				if ('original' in self.benchmark_scores.keys())&('shuffled' in self.benchmark_scores.keys()):
					df_benchmark_scores_original = pd.DataFrame(data=self.benchmark_scores['original'])
					df_benchmark_scores_shuffled = pd.DataFrame(data=self.benchmark_scores['shuffled'])
					n_benchmark_original = len(df_benchmark_scores_original)
					n_benchmark_shuffled = len(df_benchmark_scores_shuffled)
					print(f'• Original ({n_benchmark_original} tests) :')
					print(df_benchmark_scores_original)
					print(f'• Shuffled ({n_benchmark_shuffled} tests) :')
					print(df_benchmark_scores_shuffled)
			if do_print_rule_mining_results:
				# rule_mining_results
				if r==None:
					print('\nrule_mining_results :')
					print('Number of rules :',self.ruleset_count())
				if separation_width_between_rules:
					if do_print_dataset_metadata:
						print('\n'+separation_width_between_rules*'-')
				elif r>1:
					print(f'Top {r} rules :')
				if self.ruleset_count():
					range_i = self.get_range_i()
					if r!=None:
						if r>0:
							range_i = range_i[:r]
					for i in range_i:
						if (r==None):
							print(f'\n• Rule {i} :')
							indentation = '\t'
						elif r==1:
							indentation = ''
						else:
							print(f'\n• Rule {i} :')
							indentation = '\t'
						self.i_to_print(
							i                                = i,
							indentation                      = indentation,
							do_print_rule_DataFrame          = do_print_rule_DataFrame,
							do_print_subrules_S              = do_print_subrules_S,
							do_show_cohen_d                  = do_show_cohen_d,
							do_show_wy_ratio                 = do_show_wy_ratio,
							do_show_coverage_diff            = do_show_coverage_diff,
							do_print_feature_contributions_S = do_print_feature_contributions_S,
						)
						if separation_width_between_rules>0:
							print('\n'+separation_width_between_rules*'-')
				else:
					print('No rule to show.')
	def print_light(
		self,
		print_format:str = 'list', # 'list' or 'compact'
	)->None:
		"""
		This method does a 'light' print of the InsightSolver.
	
		Two formats:
		- 'list': shows the rules via a loop of prints.
		- 'compact': shows the rules in a single DataFrame.
		"""
		with pd.option_context('display.max_columns', None, 'display.max_colwidth', 50, 'display.width', 1000):
			# Take the list of rules keys in the InsightSolver
			range_i = self.get_range_i()
			# Look at how many rules there are
			if len(range_i)==0:
				print("There are no rules in the InsightSolver.")
			else:
				print("----- Rules performance -----\n")
				# Handle the rules performance.
				d_i_scores = {
					'i' : range_i,
				}
				keys = [
					'p_value',
					'F_score',
					'Z_score',
					'coverage',
					'm1/M1',
					'mu_rule',
					'lift',
					'complexity_S',
				]
				for key in keys:
					d_i_scores[key] = [self.rule_mining_results[i][key] for i in range_i]
				# Convert the dict to a DataFrame
				df_i_scores = pd.DataFrame(d_i_scores)
				# Limit the number of digits shown for the p-value
				df_i_scores['p_value'] = df_i_scores['p_value'].apply(lambda x:format_value(value=x,format_type='scientific',decimals=6))
				# Rename the complexity
				df_i_scores.rename(columns={'complexity_S':'complexity'},inplace=True)
				# Set 'i' as an index
				df_i_scores.set_index('i',inplace=True)
				# Show the result
				print(df_i_scores)

				print("\n------- Rules details -------\n")
				# Handle the rules details
				if print_format=='compact':
					l_df = []
				for i_rule in range_i:
					if print_format=='list':
						if i_rule==0:
							print(f'i={i_rule}:')
						else:
							print(f'\ni={i_rule}:')

					rule_i = self.i_to_rule(i=i_rule)
					# Take the DataFrame of feature contributions
					df_feature_contributions_S = self.i_to_feature_contributions_S(i=i_rule,do_ignore_col_rule_S=False)
					df_feature_contributions_S['description'] = df_feature_contributions_S.index.map(self.columns_descr).fillna('')
					df_feature_contributions_S = df_feature_contributions_S[['description','rule_S','p_value']]
					df_feature_contributions_S.rename(columns={'rule_S':'rule','p_value':'contribution'},inplace=True)
					if print_format=='compact':
						df_feature_contributions_S.reset_index(inplace=True)
						df_feature_contributions_S['i'] = i_rule
						l_df.append(df_feature_contributions_S)
					elif print_format=='list':
						print(df_feature_contributions_S)
				if print_format=='compact':
					df = pd.concat(
						objs         = l_df,
						axis         = 0,
						ignore_index = True,
					)
					df = df[['i','feature_name','description','rule','contribution']]
					print(df)
				print("\n-----------------------------")
	def print_dense(
		self,
	)->None:
		"""
		This method is aimed at printing a 'dense' version of the InsightSolver object.
		"""
		with pd.option_context('display.max_columns', 10, 'display.max_colwidth', 100, 'display.width', 1000):
			# Take the list of rules keys in the InsightSolver object.
			range_i = self.get_range_i()
			# Look at how many rules there are.
			if len(range_i)==0:
				print("There are no rules in the InsightSolver.")
			else:
				# Take the performance scores of the rules.
				d_i_scores = {
					'i' : range_i,
				}
				for key in [
					'p_value',
					'coverage',
					'lift',
					'rule_S',
				]:
					d_i_scores[key] = [self.rule_mining_results[i][key] for i in range_i]
				df_i_scores = pd.DataFrame(d_i_scores)
				l_df_temp = []
				for i in range_i:
					rule_S = df_i_scores.loc[i,'rule_S']
					s_temp = pd.Series(rule_S,name='rule')
					s_temp.index.name = 'variable'
					# Add an empty row to make the view cleaner
					s_space = pd.Series(
						data  = [""],
						index = pd.Series([""],name='variable'),
						name  = 'rule',
					)
					s_temp = pd.concat([s_space, s_temp],axis=0)
					df_temp = s_temp.to_frame().reset_index()
					# Add some informations
					df_temp['i']        = i
					df_temp['p_value']  = self.rule_mining_results[i]['p_value']
					df_temp['coverage'] = self.rule_mining_results[i]['coverage']
					df_temp['lift']     = self.rule_mining_results[i]['lift']
					df_temp['']         = ''
					l_df_temp.append(df_temp)
				# Concatenate the DataFrames
				df_concat = pd.concat(l_df_temp,axis=0,ignore_index=True)
				# Add a column "contribution"
				for i in range_i:
					rule_S = df_i_scores.loc[i,'rule_S']
					for variable in rule_S.keys():
						contribution = self.rule_mining_results[i]['feature_contributions_S']['p_value_contribution'][variable]
						mask  = df_concat['i'] == i
						mask &= df_concat['variable'] == variable
						df_concat.loc[mask, 'contribution'] = contribution
				df_concat = df_concat.sort_values(by=['i', 'contribution'], ascending=[True, False],na_position='first')
				# Manage the column 'contribution'
				#df_concat['contribution'] = df_concat['contribution'].apply(lambda x: '' if pd.isna(x) else f"{x:.2f}")
				#df_concat['contribution'] = df_concat['contribution'].apply(lambda x: '' if pd.isna(x) else f"{x * 100:.1f}%")
				df_concat['contribution'] = df_concat['contribution'].apply(lambda x: format_value(value=x,format_type='percentage',decimals=1))
				# Handle the rule's behaviour for NaNS
				df_concat['nans'] = ''
				for i in range_i:
					rule_S = df_i_scores.loc[i,'rule_S']
					for variable in rule_S.keys():
						rule = rule_S[variable]
						if isinstance(rule,list):
							if len(rule)==2:
								if rule[1] in ['exclude_nan','include_nan']:
									mask  = df_concat['i'] == i
									mask &= df_concat['variable'] == variable
									rule,nan = rule
									df_concat.loc[mask, 'nans'] = nan.split('_')[0]
									df_concat.loc[mask, 'rule'] = pd.Series([rule], index=df_concat.index[mask])
				# Hangle the column 'lift'
				df_concat['lift'] = df_concat['lift'].round(2)
				# Hangle the column 'coverage'
				#df_concat['coverage'] = df_concat['coverage'].round(3)
				#df_concat['coverage'] = df_concat['coverage'].apply(
				#	lambda x: '' if pd.isna(x) else f"{x * 100:.1f}%"
				#)
				df_concat['coverage'] = df_concat['coverage'].apply(lambda x:format_value(value=x,format_type='percentage',decimals=1))
				df_concat['coverage'] = df_concat['coverage'].apply(
					lambda x: ' '+x if float(x[:-1])<10 else x
				)
				# Handle the column 'p_value'
				#df_concat['p_value'] = df_concat['p_value'].apply(lambda x: f"{x:.2e}")
				#df_concat['p_value'] = df_concat['p_value'].apply(lambda x: f"{x:.0e}")
				df_concat['p_value'] = df_concat['p_value'].apply(lambda x: format_value(value=x,format_type='scientific_no_decimals'))
				# Put certain columns in the index
				cols_index = [
					'i',
					'p_value',
					'coverage',
					'lift',
					''
				]
				df_concat.set_index(cols_index,inplace=True)
				# Reorder some columns
				cols = ['contribution','variable','rule','nans']
				df_concat = df_concat[cols]
				# Format the DataFrame even more
				df_string = df_concat.to_string()
				lines = df_string.split('\n')
				index_columns = lines[0]        # Name of the index columns
				header = lines[1]               # Name of the main columns
				content = "\n".join(lines[2:])  # Main content
				df_string_with_spacing = f"{index_columns}\n{header}\n\n{content}"
				# Show the result
				print(df_string_with_spacing)
	def to_dict(
		self,
	):
		"""
		This method aims to export the content of the InsightSolver object to a dictionary.
		"""
		from copy import deepcopy
		# Declare a Python dictionary
		d = dict()
		# dataset_metadata :
		d['dataset_metadata']                                       = dict()
		d['dataset_metadata']['target_threshold']                   = self.target_threshold
		d['dataset_metadata']['M']                                  = self.M
		d['dataset_metadata']['M0']                                 = self.M0
		d['dataset_metadata']['M1']                                 = self.M1
		d['dataset_metadata']['columns_names_to_descr']             = self.columns_descr
		d['dataset_metadata']['features_names_to_other_modalities'] = self.other_modalities
		# monitoring_metadata
		d['monitoring_metadata']                                    = deepcopy(self.monitoring_metadata)
		# benchmark_scores
		d['benchmark_scores']                                       = deepcopy(self.benchmark_scores)
		# rule_mining_results :
		d['rule_mining_results']                                    = deepcopy(self.rule_mining_results)
		# Return the result
		return d
	def to_json_string(
		self,
		verbose      = False,
	):
		"""
		This method aims to export the content of the InsightSolver object to a JSON string.
		"""
		# Export the InsightSolver object to a dict
		d = self.to_dict()
		# Convert the dict to a JSON string
		import jsonpickle # pip install jsonpickle
		json_string = jsonpickle.encode(d)
		# Return the result
		return json_string
	def to_dataframe(
		self,
		verbose            = False,
		do_append_datetime = False,
		do_rename_cols     = False,
	):
		"""
		This method aims to export the content of the InsightSolver object to a DataFrame.
		"""
		# Handling the rules
		if verbose:
			print('Creating df_rule_mining_results...')
		df = pd.DataFrame.from_dict(
			data   = self.rule_mining_results,
			orient = 'index',
		)
		df.index.name = 'i'
		df.reset_index(inplace=True)
		cols_rule_mining_results = df.columns.to_list()
		if verbose:
			print(df)
		# Handling the metadata
		cols_metadata = [
			'target_name',                   # Dataset metadata     - str
			'target_goal',                   # Dataset metadata     - str / int
			'target_threshold',              # Dataset metadata     - int / float
			'M',                             # Dataset metadata     - int
			'M0',                            # Dataset metadata     - int
			'M1',                            # Dataset metadata     - int
			'columns_descr',                 # Dataset metadata     - dict
			'specified_constraints',         # Constraints medatada - dict
			'benchmark_scores',              # Benchmarking scores  - dict
		]
		for col_metadata in cols_metadata:
			df[col_metadata] = len(df)*[getattr(self, col_metadata)]
		# If we want to add a datetime column to specify when the table was created
		if do_append_datetime:
			from datetime import datetime
			df['datetime_export'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		# Order the columns
		cols_A = [
			'datetime_export',
			'user_id',
			'target_name',
			'target_goal',
			'target_threshold',
			'M',
			'M0',
			'M1',
			'columns_descr',
			'specified_constraints',
			'benchmark_scores',
			'i',
			'm',
			'm0',
			'm1',
			'coverage',
			'm1/M1',
			'mu_rule',
			'mu_pop',
			'sigma_pop',
			'lift',
			'p_value',
			'F_score',
			'Z_score',
			'rule_S',
			'complexity_S',
			'F1_pop',
			'G_bad_class',
			'G_information',
			'G_gini',
			'p_value_ratio_S',
			'F_score_ratio_S',
			'subrules_S',
			'feature_contributions_S',
			'shuffling_scores'
		]
		cols_B = df.columns.to_list()
		cols_B_minus_A = [col for col in cols_B if col not in cols_A]
		if len(cols_B_minus_A)>0:
			raise Exception(f"ERROR: Some columns are missing in the implementation : {cols_B_minus_A}")
		cols_A_inter_B = [col for col in cols_A if col in cols_B]
		df = df[cols_A_inter_B]
		# Rename some columns
		if do_rename_cols:
			"""
			This renaming is useful for BigQuery:
			- Forbidden to have '/' in a column name
			- Columns names are not case sensitive, so it cannot distinguish between M and m, M0 and m0, M1 and m1.
			"""
			d_rename = {
				'm1/M1' : 'coverage1', # To avoid the character '/'
				'M'     : 'm_pop',     # To avoir the collision between M and m
				'M0'    : 'm0_pop',    # To avoir the collision between M0 and m0
				'M1'    : 'm1_pop',    # To avoir the collision between M1 and m1
				'm'     : 'm_rule',    # To avoir the collision between M  and m
				'm0'    : 'm0_rule',   # To avoir the collision between M0 and m0
				'm1'    : 'm1_rule',   # To avoir the collision between M1 and m1
			}
			df.rename(
				columns = d_rename,
				inplace = True,
			)
		# Return the result
		return df
	def to_csv(
		self,
		output_file    = None,
		verbose        = False,
		do_rename_cols = False,
	):
		"""
		This method is meant to export the content of the InsightSolver object to a CSV file.
		"""
		# Avoid to generate a string containing np.float64 and np.int64 everywhere
		if np.__version__>='2.0.0':
			np.set_printoptions(legacy='1.25')

		df = self.to_dataframe(
			do_rename_cols = do_rename_cols,
		)
		if (output_file!=None)&verbose:
			print('Exporting :',output_file)

		# Make sure that np.int64 and np.float64 are not written everywhere.
		# Create the CSV string
		csv_string = df.to_csv(output_file,index=False)
		if (output_file!=None)&verbose:
			print('Done.')
		# Return the result
		return csv_string

################################################################################
################################################################################
# Defining the API Client

def search_best_ruleset_from_API_public(
	df                      : pd.DataFrame,                                        # The Pandas DataFrame that contains the data to analyse.
	computing_source        : str                                        = 'auto', # Specify if the execution is local or remote
	input_file_service_key  : Optional[str]                              = None,   # For a remote execution from outside GCP, a service key file is necessary
	user_email              : Optional[str]                              = None,   # For a remote execution from inside GCP, a user email is necessary
	target_name             : Optional[Union[str,int]]                   = None,   # Name of the target variable
	target_goal             : Optional[Union[str,numbers.Real,np.uint8]] = None,   # Target goal
	columns_names_to_btypes : Optional[Dict]                             = dict(), # Specify the btypes of the variables
	threshold_M_max         : Optional[int]                              = None,   # Specify the maximum number of rows to use in the rule mining
	specified_constraints   : Optional[Dict]                             = dict(), # Specify some constraints on the rules
	top_N_rules             : Optional[int]                              = 10,     # Maximum number of rules to keep
	verbose                 : bool                                       = False,  # Verbosity
	filtering_score         : str                                        = 'auto', # Filtering score
	api_source              : str                                        = 'auto', # Source of the API call
	do_compress_data        : bool                                       = True,   # If we want to compress the communications (slower to compress but faster to transmit)
	do_compute_memory_usage : bool                                       = True,   # If we want to compute the memory usage of the API (this significantly slows down computation time but is good for monitoring purposes)
	n_benchmark_original    : int                                        = 5,      # Number of benchmarking runs to execute where the target is not shuffled.
	n_benchmark_shuffle     : int                                        = 20,     # Number of benchmarking runs to execute where the target is shuffled.
)->dict:
	"""
	This function is meant to make a rule mining API call.

	Parameters
	----------
	df: DataFrame
		The DataFrame that contains the data to analyse (a target column and various feature columns).
	computing_source: str
		If the rule mining should be computed locally or remotely.
	input_file_service_key: str
		The string that specifies the path to the service key (necessary to use the remote Cloud Function from outside GCP).
	user_email: str
		The email of the user (necessary to use the remote Cloud Function from inside GCP).
	target_name: str
		Name of the column of the target variable.
	target_goal: str (or other modality of the target variable)
		Target goal.
	columns_names_to_btypes: dict
		A dict that specifies the btypes of the columns.
	threshold_M_max: int
		Threshold on the maximum number of points to use during the rule mining (max. 10000 pts in the public API).
	specified_constraints: dict
		A dict that specifies contraints to be used during the rule mining.
	top_N_rules: int
		An integer that specifies the maximum number of rules to get from the rule mining.
	verbose: bool
		Verbosity.
	filtering_score: str
		A string that specifies the filtering score to be used when selecting rules.
	api_source: str
		A string used to identify the source of the API call.
	do_compress_data: bool
		A boolean that specifies if we want to compress the data.
	do_compute_memory_usage: bool
		A bool that specifies if we want to get the memory usage of the computation.
	n_benchmark_original: int
		Number of benchmarking runs to execute where the target is not shuffled.
	n_benchmark_shuffle: int
		Number of benchmarking runs to execute where the target is shuffled.

	Returns
	-------
	response: requests.models.Response
		A response object obtained from the API call that contains the rule mining results.
	"""
	# Manage where the computation is executed
	if computing_source=='auto':
		computing_source='remote_cloud_function'
	# Taking the global variables
	if api_source=='auto':
		api_source = API_SOURCE_PUBLIC
	# Manage the btypes
	if columns_names_to_btypes==None:
		columns_names_to_btypes = dict()
	# Manage the specified constraints
	if specified_constraints==None:
		specified_constraints = dict()
	# Manage top_N_rules
	if top_N_rules==None:
		top_N_rules = 10
	# Convert the Pandas DataFrame to JSON
	df_to_dict = df.to_json()
	# Create a dict that contains relevant informations for rule mining
	d_out_original = {
		'df_to_dict'              : df_to_dict,
		'target_name'             : target_name,
		'target_goal'             : target_goal,
		'columns_names_to_btypes' : columns_names_to_btypes,
		'threshold_M_max'         : threshold_M_max,
		'specified_constraints'   : specified_constraints,
		'top_N_rules'             : top_N_rules,
		'filtering_score'         : filtering_score,
		'api_source'              : api_source,
		'n_benchmark_original'    : n_benchmark_original,
		'n_benchmark_shuffle'     : n_benchmark_shuffle,
	}
	# Make the API call
	from .api_utilities import search_best_ruleset_from_API_dict
	d_in_original = search_best_ruleset_from_API_dict(
		d_out_original          = d_out_original,
		input_file_service_key  = input_file_service_key,
		user_email              = user_email,
		computing_source        = computing_source,
		do_compress_data        = do_compress_data,
		do_compute_memory_usage = do_compute_memory_usage,
		verbose                 = verbose,
	)
	# Return the result
	return d_in_original

################################################################################
################################################################################
