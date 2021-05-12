from setuptools import setup, find_packages

requirements = [
    "sklearn",
    "numpy",
    "numba",
    "joblib",
    "scipy",
]

with open("README.md", mode="r", encoding = "utf8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="aggclass",
    version="0.0.1",
    author="Hayden Helm",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/hhelm10/agg-class/",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    install_requires=requirements,
)
