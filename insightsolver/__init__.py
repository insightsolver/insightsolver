"""
* `Organization`:  InsightSolver
* `Project Name`:  InsightSolver
* `Module Name`:   insightsolver
* `File Name`:     __init__.py
* `Author`:        Noé Aubin-Cadot
* `Email`:         noe.aubin-cadot@insightsolver.com
* `Last Updated`:  2025-04-23
* `First Created`: 2024-09-16

Description
-----------
The Python module `insightsolver` is an API client of the InsightSolver SaaS which is designed to generate advanced rule mining and data insights.

License
-------
Exclusive Use License - see `LICENSE <license.html>`_ for details.

----------------------------

"""

from .insightsolver import InsightSolver
from .version import __version__

__all__ = [
	"InsightSolver", # On ne rend accessible que la classe InsightSolver
]