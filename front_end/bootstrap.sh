#!/bin/sh
export FLASK_APP=./src/app.py
export PYTHONPATH=./
flask run -h 0.0.0.0 --port 5050