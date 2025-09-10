from setuptools import setup, find_packages

# Read the file README in UTF-8
with open("README.md", "r", encoding="utf-8") as fh:
	long_description = fh.read()

def read_version():
	import re
	with open("insightsolver/version.py", encoding="utf-8") as f:
		match = re.search(r'^__version__ = ["\']([^"\']+)["\']', f.read())
		if match:
			return match.group(1)
		raise RuntimeError("Unable to find version string.")

# Read requirements.txt and ignore ignorer the rows that start with git
with open("requirements.txt") as f:
	requirements = [
		line.strip()
		for line in f
		if line.strip() and not line.startswith("git+")
	]

setup(
	name="insightsolver",                   # Name of the package (must be unique on PyPI)
	version=read_version(),                 # Version of the package
	packages=find_packages(),               # Find all sub-packages automatically
	install_requires=requirements,
	# Main dev or team that created the project
	author="Noé Aubin-Cadot",
	author_email="noe.aubin-cadot@insightsolver.com",
	# Maintainer of the project
	maintainer="InsightSolver Solutions Inc.",
	maintainer_email="support@insightsolver.com",
	description="InsightSolver offers rule-based insights generation for actionable data-driven decisions.",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/insightsolver/insightsolver",  # URL of the project
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: Other/Proprietary License",
		"Operating System :: OS Independent",
	],
	python_requires='>=3.9',
	include_package_data=True,
	package_data={
		"insightsolver": ["assets/*.png"],
	},
)