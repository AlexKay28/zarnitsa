from setuptools import setup, find_packages

setup(
    name='zarnitsa',
    packages=find_packages(),
    package_data={'zarnitsa': ['internal_data/*']},
    description='zarnitsa package for data augmentation',
    version='0.0.1',
    url='https://github.com/AlexKay28/zarnitsa',
    author='Kaigorodov Alexander',
    author_email='kaygorodo2305@gmail.com',
)
