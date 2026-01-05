import json
import re

input_path = "/Users/harunisik/.gemini/antigravity/brain/4103a2f2-bb0f-4780-8001-a0d700fff9e7/google_collab_export.txt"
output_path = "/Users/harunisik/Desktop/ChatBot-Optimization/ChatBot_Optimization.ipynb"

with open(input_path, "r", encoding="utf-8") as f:
    content = f.read()

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        },
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

# Split by CELL markers
lines = content.split('\n')
current_cell = []
cell_start = False

for line in lines:
    if line.startswith("# --- CELL"):
        # Save previous cell
        if current_cell:
            cell_content = '\n'.join(current_cell)
            if cell_content.strip():
                notebook["cells"].append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [cell_content]
                })
        current_cell = []
        cell_start = True
    elif cell_start or current_cell:
        current_cell.append(line)

# Add last cell
if current_cell:
    cell_content = '\n'.join(current_cell)
    if cell_content.strip():
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [cell_content]
        })

# Write
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"✅ Created notebook with {len(notebook['cells'])} cells")
print(f"Output: {output_path}")
