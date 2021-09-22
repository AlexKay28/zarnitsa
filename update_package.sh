#!/bin/bash
python3 -m pip install --upgrade build
python3 -m build

python3 -m pip install --upgrade twine

python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/* --verbose
