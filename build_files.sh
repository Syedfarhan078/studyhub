#!/bin/bash

# 1. Install dependencies
echo "Installing project dependencies..."
python3 -m pip install -r requirements.txt

# 2. Collect static files for WhiteNoise
echo "Collecting static files..."
python3 manage.py collectstatic --noinput --clear

# 3. Apply database migrations to Neon Cloud PostgreSQL
echo "Applying database migrations to Neon..."
python3 manage.py migrate --noinput

# 4. Automatically inject all 5 VTU subjects and topics into Neon!
echo "Loading curriculum data into Neon..."
python3 manage.py loaddata curriculum_data.json

echo "Build completed successfully!"