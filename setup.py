from setuptools import find_packages, setup
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


with open(path.join(HERE, "requirements.txt"), "r") as f:
    requirements = f.read().splitlines()

setup(
    name='socialysis',
    packages=find_packages(include=['socialysis']),
    include_package_data=True,
    version='0.0.1',
    url="https://github.com/abdulrahmankhayal/socialysis",
    description='Tool for analyzing and extracting insights from Facebook Messenger conversations',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Abdul-Rahman Khayyal',
    license='MIT',
    keywords="Facebook Messenger Analysis",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    package_data={
        '': ['./stickers_used/*','./sw.txt','SAMPLE_DATA.pkl','sample_meta_data.json']
    }
)
