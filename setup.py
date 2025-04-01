from setuptools import setup, find_packages

setup(
    name="bnm",
    version="0.1.0",
    description="A package for evaluating and visualizing DAGs.",
    long_description=open("README.md", encoding="utf-8").read(),   
    long_description_content_type="text/markdown",
    author="Pavel Averin",
    author_email="averinjr@gmail.com",
    url="https://github.com/averinpa/bnm",
    packages=find_packages(),
    install_requires=[
        "networkx",
        "graphviz",
        "pandas",
        "numpy",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    include_package_data=True,
)
