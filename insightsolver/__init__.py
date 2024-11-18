"""
Fichier __init__.py
Ici on détermine ce qui est rendu public.
On ne va rendre public que la classe InsightSolver car ses méthodes utilisent les autres fonctions.

Pour appeler la classe on fait :
	from insightsolver import InsightSolver


"""

from .insightsolver import InsightSolver

__all__ = [
	"InsightSolver", # On ne rend accessible que la classe InsightSolver
]