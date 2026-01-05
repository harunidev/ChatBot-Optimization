
import json
import re
import os

# Define paths
input_path = "/Users/harunisik/.gemini/antigravity/brain/4103a2f2-bb0f-4780-8001-a0d700fff9e7/google_collab_export.txt"
output_path = "/Users/harunisik/.gemini/antigravity/brain/4103a2f2-bb0f-4780-8001-a0d700fff9e7/ChatBot_Optimization.ipynb"

# Read the export file
with open(input_path, "r", encoding="utf-8") as f:
    content = f.read()

# Define the notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        },
        "colab": {
            "provenance": []
        },
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

# Split content by cell markers
# Regex to find lines like "# --- CELL X: Title ---"
cell_pattern = re.compile(r'(# --- CELL \d+: .*? ---)')
parts = cell_pattern.split(content)

# The first part is usually header comments before the first cell
if parts[0].strip():
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in parts[0].strip().split("\n")]
    })

# Iterate through parts (header is idx 0, then (marker, content), (marker, content)...)
for i in range(1, len(parts), 2):
    marker = parts[i]
    if i + 1 < len(parts):
        cell_content = parts[i+1]
        
        # Determine strict code lines (removing leading/trailing empty lines but keeping structure)
        lines = cell_content.strip().split("\n")
        
        # Add the marker as a comment at the top of the cell
        source_lines = [marker + "\n"] + [line + "\n" for line in lines]
        
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source_lines
        })

# Write the notebook file
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Successfully converted {input_path} to {output_path}")
