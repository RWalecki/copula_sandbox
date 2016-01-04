from setuptools import setup, find_packages
import os

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "Multivariate_Sampling",
    version = "0.0.1",
    author = "Robert Walecki",
    author_email = "r.walecki14@imperial.ac.uk",
    description = (""),
    license = "BSD",
    keywords = "",
    url = "",
    packages=find_packages()
)
