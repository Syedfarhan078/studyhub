#!/usr/bin/env bash
# Exit on error
set -o errexit

# 1. Install production packages
pip install -r requirements.txt

# 2. Bundle static files for WhiteNoise
python manage.py collectstatic --no-input

# 3. Apply database migrations to PostgreSQL
python manage.py migrate

python manage.py loaddata curriculum_seed.json