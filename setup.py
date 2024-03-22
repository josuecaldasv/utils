from setuptools import setup, find_packages

setup(
    name="utils",
    version="0.1",
    packages=find_packages(),
    description="Utilidades  para proyectos de Python",
    author="Josue Caldas",
    author_email="tu_email@example.com",
    url="https://github.com/josuecaldasv/utils",
    install_requires=[

        "mlxtend",
        "scikit-learn"

    ],
)
