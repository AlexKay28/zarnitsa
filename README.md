<p align="center">
  <img src="https://user-images.githubusercontent.com/55444371/126209354-44068bb7-81aa-49a5-af4e-71b8c2475386.png" />
</p>

![GitHub](https://img.shields.io/github/license/heartexlabs/label-studio?logo=heartex) [![Wheel](https://img.shields.io/pypi/wheel/textaugment.svg?maxAge=3600)](https://pypi.python.org/pypi/textaugment)  [![python](https://img.shields.io/pypi/pyversions/textaugment.svg?maxAge=3600)](https://pypi.org/project/textaugment/)
# zarnitsa package

Zarnitsa package for data augmentation techniques.

- Internal data augmentation using existed data
- External data augmentation setting known statistical distributions by yourself
- NLP augmentation


## Principal scheme of project (currently)

![Screenshot_20210722_232750](https://user-images.githubusercontent.com/55444371/126705231-3a052e84-6a9c-4c1a-a772-5caa8c2e7c4b.png)

## Requirements
- Python3
- numpy
- pandas
- nlpaug
- wget
- scikt-learn

## Installation
Install package using PyPI:
```
pip install zarnitsa
```

## Usage
Simple usage examples:
### Augmentation internal.
This is type of augmentation which you may use in case of working with numerical features.
```
>>> from zarnitsa.DataAugmenterInternally import DataAugmenterInternally

>>> daug_comb = DataAugmenterInternally()
>>> aug_types = [
>>>     "normal",
>>>     "uniform",
>>>     "permutations",
>>> ]

>>> # pd Series object example
>>> s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
>>> for aug_type in aug_types:
>>>     print(aug_type)
>>>     print(daug_comb.augment_column(s, freq=0.5, return_only_aug=True, aug_type=aug_type))
normal
7     9.958794
3     0.057796
0    -3.135995
6     7.197400
8    13.258087
dtype: float64
uniform
2    10.972232
8     5.335357
9     9.111281
5     5.964971
4    -0.210732
dtype: float64
permutations
4     6
5     4
9    10
3     3
2     5
dtype: int64
```

### Augmentation NLP
This is type of augmentation which you may use in case of working with textual information.

```
>>> from zarnitsa.DataAugmenterNLP import DataAugmenterNLP

>>> daug = DataAugmenterNLP()

>>> text = "This is sentence example to augment. Thank you for interesting"

>>> daug.augment_column_wordnet(text)
'This be sentence example to augment. Thank you for concern'

>>> daug.augment_column_del(text, reps=1)
'This is sentence example to augment. you for interesting'

>>> daug.augment_column_permut(text, reps=1)
'This sentence is example to augment. Thank you for interesting'
```
### Augmentation External
This is type of augmentation which you may use in case of working with distribution modeling
having prior knowlege about it

_Will be soon..._

