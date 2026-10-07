# Code Ocean compute environment for ACCESS3731935
# Matches the environment reported in the paper (Python 3.12, NumPy 2.3.5).
FROM python:3.12-slim

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Code Ocean mounts /code, /data, and /results at run time.
