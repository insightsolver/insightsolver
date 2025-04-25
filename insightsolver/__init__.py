"""
* `Organization`:  InsightSolver
* `Project Name`:  InsightSolver
* `Module Name`:   insightsolver
* `File Name`:     __init__.py
* `Author`:        Noé Aubin-Cadot
* `Email`:         noe.aubin-cadot@insightsolver.com
* `Last Updated`:  2025-04-24
* `First Created`: 2024-09-16

Description
-----------
The Python module `insightsolver` is an API client of the InsightSolver SaaS which is designed to generate advanced rule mining and data insights.

License
-------
Exclusive Use License - see `LICENSE <license.html>`_ for details.

----------------------------

"""

# Import the version of the module
from .version import __version__

__all__ = [
	"InsightSolver", # On ne rend accessible que la classe InsightSolver
]

def __getattr__(name):
	"""
	This function is meant to make a differed import of the class InsightSolver after the pip install.
	Without it, the pip install tries to import Pandas earlier than it is installed.
	"""
	if name == "InsightSolver":
		from .insightsolver import InsightSolver
		return InsightSolver
	raise AttributeError(f"module {__name__} has no attribute {name}")