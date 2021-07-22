from setuptools import setup, find_packages

install_requires = [
    'numpy',        # numpy==1.18.5
    'pandas',       # pandas==1.1.5
    "scikit-learn", # scikit-learn==0.24.1
    "nlpaug",       # nlpaug==1.1.3
    "wget",         # wget==3.2
]

setup(
    name='zarnitsa',
    packages=find_packages(),
    package_data={'zarnitsa': ['internal_data/*']},
    description='zarnitsa package for data augmentation',
    version='0.0.1',
    url='https://github.com/AlexKay28/zarnitsa',
    author='Kaigorodov Alexander',
    author_email='kaygorodo2305@gmail.com',
    download_url='https://pypi.org/project/zarnitsa/',
    install_requires=install_requires,
    keywords=['augmentation', 'NLP', 'distributions'],
    include_package_data=True
)
