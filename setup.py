from setuptools import setup, find_packages

setup(
    name="utils",
    version="0.1",
    packages=find_packages(),
    description="Tools for Python programming",
    author="Josue Caldas",
    author_email="josue.caldas@pucp.edu.pe",
    url="https://github.com/josuecaldasv/utils",
    install_requires=[

        "mlxtend",
        "scikit-learn"

    ],
)
