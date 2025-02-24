from setuptools import setup, find_packages

# Lire le fichier README en utf-8
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
	name="insightsolver",                   # Nom du package (doit être unique sur PyPI)
	version="0.1.8",                        # Version du package
	packages=find_packages(),               # Trouver tous les sous-packages automatiquement
	install_requires=[                      # Dépendances du package
		"pandas",
		"numpy",
		"requests",
		"mpmath",
		"jsonpickle",
		"cryptography",
		"google-auth",
	],
	# Développeur principal ou équipe qui a créé le projet
	author="Noé Aubin-Cadot",
	author_email="noe.aubin-cadot@insightsolver.com",
	# Organisation responsable de la maintenance du projet
    maintainer="InsightSolver",
    maintainer_email="support@insightsolver.com",
	description="InsightSolver offers rule-based insights generation for actionable data-driven decisions.",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/insightsolver/insightsolver",  # URL du projet
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: Other/Proprietary License",
		"Operating System :: OS Independent",
	],
	python_requires='>=3.9',
	include_package_data=True,
)