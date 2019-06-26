from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()


with open("LICENSE.md") as f:
    license = f.read()


setup(
    name="sparsecomputation",
    version="2019.6.1",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.5",
    ],
    description="Sparse Computation algorithm for sparsifying similarity matrices.",
    keywords="pairwise similarity, classification, sparsification," + "clustering",
    url="https://github.com/hochbaumGroup/sparsecomputation",
    author="Titouan Jehl, Quico Spaen, Philipp Baumann",
    author_email="qspaen@berkeley.edu",
    license=license,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["numpy", "scipy", "sklearn", "six"],
)
