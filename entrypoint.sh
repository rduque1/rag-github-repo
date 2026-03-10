#!/bin/sh

# Create tables in the database
uv run -m src.core.database

# Start the application
uv run -m streamlit run src/interface.py
