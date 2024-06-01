"""
Setup script for the Predictive Workplace Safety Analytics Platform
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
try:
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    print("Warning: requirements.txt not found")

setup(
    name="workplace-safety-analytics",
    version="1.0.0",
    author="Workplace Safety Analytics Team",
    author_email="safety-analytics@example.com",
    description="Predictive analytics platform for workplace safety using OSHA data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/workplace-safety-analytics",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "safety-analytics-download=data.download_osha_data:main",
            "safety-analytics-process=data.process_data:main",
            "safety-analytics-train=models.train_model:main",
            "safety-analytics-init-db=database.init_db:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.sql"],
    },
    keywords=[
        "workplace safety",
        "OSHA",
        "machine learning",
        "predictive analytics",
        "risk assessment",
        "ensemble models",
        "safety management"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/workplace-safety-analytics/issues",
        "Source": "https://github.com/your-username/workplace-safety-analytics",
        "Documentation": "https://workplace-safety-analytics.readthedocs.io/",
    },
) 