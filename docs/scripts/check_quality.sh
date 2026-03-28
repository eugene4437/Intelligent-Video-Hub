#!/bin/bash
# Перевірка типів та лінтинг
mypy src/
flake8 src/
echo "Quality check finished"
