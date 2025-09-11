from setuptools import setup, find_packages
import os
import re

# Read the file README in UTF-8
with open("README.md", "r", encoding="utf-8") as fh:
	long_description = fh.read()

def read_version():
	version_path = os.path.join(os.path.dirname(__file__), "insightsolver", "version.py")
	print(f"[*] Trying to open: {version_path}")
	if not os.path.exists(version_path):
		raise RuntimeError(f"Version file not found at {version_path}")
	with open(version_path, encoding="utf-8") as f:
		content = f.read()
		print(f"[*] File opened, content:\n{content}")
		
		#match = re.search(r'^__version__ = ["\']([^"\']+)["\']', f.read())
		match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', content, re.M)
		if match:
			print(f"[*] Found version: {match.group(1)}")
			return match.group(1)
		raise RuntimeError("Unable to find version string in the version file.")

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