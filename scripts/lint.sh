#!/bin/bash
echo "Running Pylint..."
pylint src/ main.py
echo "Running Mypy..."
mypy src/ main.py
