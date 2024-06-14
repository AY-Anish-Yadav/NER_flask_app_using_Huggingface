from setuptools import find_packages,setup

setup(
    name='NER',
    version='0.0.1',
    author='Anish Yadav',
    author_email='reach.anish.yadav@gmail.com',
    install_requires=["flask","transformers"],
    packages=find_packages()
)