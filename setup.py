from setuptools import setup, find_packages

install_requires = [
    "numpy==1.18.5",
    "pandas==1.1.5",
    "scikit-learn==0.24.1",
    "nlpaug==1.1.3",
    "wget==3.2",
    "spacy==3.0.5",
]

setup(
    name="zarnitsa",
    packages=find_packages(),
    package_data={"zarnitsa": ["internal_data/*"]},
    description="zarnitsa package for data augmentation",
    version="0.0.13",
    url="https://github.com/AlexKay28/zarnitsa",
    author="Kaigorodov Alexander",
    author_email="kaygorodo2305@gmail.com",
    download_url="https://pypi.org/project/zarnitsa/",
    install_requires=install_requires,
    keywords=["augmentation", "NLP", "distributions"],
    include_package_data=True,
)
