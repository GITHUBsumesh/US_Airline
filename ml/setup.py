'''
The setup.py file is an essential part of packaging and 
distributing Python projects. It is used by setuptools 
to define the configuration of your project, such as its metadata, dependencies, and more
'''

from setuptools import find_packages,setup
from typing import List

def get_requirements()->List[str]:
    """
    This function will return list of requirements
    
    """
    requirement_lst:List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            #Read lines from the file
            lines=file.readlines()
            ## Process each line
            for line in lines:
                requirement=line.strip()
                ## ignore empty lines and -e .
                if requirement and requirement!= '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_lst

setup(
    name="US_Airline",
    version="0.0.1",
    author="Sumesh Ranjan Majee",
    author_email="sumesh1999sumesh@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)