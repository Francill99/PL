from setuptools import setup, find_packages

setup(
    name="Pseudolikelihood_Analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy", "torch"  # add other dependencies here
    ],
    python_requires=">=3.10",
)