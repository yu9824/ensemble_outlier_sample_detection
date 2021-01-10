from setuptools import setup, find_packages

with open('requirements.txt') as requirements_file:
    install_requirements = requirements_file.read().splitlines()

setup(
    name='ensemble_outlier_sample_detection',
    version='0.0.1',
    description='You can do outlier detection.',
    author='yu-9824',
    author_email='yu.9824@gmail.com',
    install_requires=install_requirements,
    url='https://github.com/yu-9824/ensemble_outlier_sample_detection',
    license=license,
    packages=find_packages(exclude=['example'])
)
