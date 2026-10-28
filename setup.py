from setuptools import setup, find_packages

setup(
    name="traffic_forecasting",
    version="0.2.0",
    description="AI-Based Network Data Analytics for 5G Traffic Forecasting",
    author="Sanchit Jadhav, Sambhav Purohit, Kunal Katkar",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "pytest>=7.4.0",
        "statsmodels>=0.14.0",
        "torch>=2.0.0",
        "flask>=2.3.0",
    ],
)
