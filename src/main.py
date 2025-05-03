# src/main.py
import typer
from pathlib import Path
from loguru import logger
from biv_mesh import BivMesh
import biv_measures
import biv_plots
from rich import print
from typing import Optional
import numpy as np
from sklearn.decomposition import PCA

app = typer.Typer(add_completion=False,
                  help="Simple tools to load, visualize, and analyze biventricular models.")
app.add_typer(biv_plots.app, name="plot")
app.add_typer(biv_measures.app)

def load_points(file_path):
    """Load control points, handling headers and zero columns"""
    with open(file_path, 'r') as f:
        next(f)  # Skip header
        data = np.loadtxt(f, delimiter=',')
    # Verify last column is all zeros before dropping it
    if not np.all(data[:, -1] == 0):
        raise ValueError(f"Last column should contain all zeros, got {data[:, -1]}")
    return data[:, :-1]  # Return only first two columns (X,Y)

@app.command(name="load")
def load_model(
    input_model: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False,
                                      help="A model txt filename."),
    model_folder: Path = typer.Option(BivMesh.DEFAULT_MODEL_FOLDER, exists=True, file_okay=False,
                                     dir_okay=True, help="Set the model folder."),
    model_name: str = typer.Option('Mesh', "-n", "--name", help="Set the model name.")
):
    """Loads a fitted model and prints the model's data structure."""
    logger.info(f"Input model file: {input_model}")
    model = BivMesh.from_fitted_model(input_model, name=model_name, model_folder=model_folder)
    print(f"There are {model.control_points.shape[0]} control points")
    print("After subdivision, here is the mesh structure:")
    print({
        'name': model.label,
        'number_of_nodes': model.nb_nodes,
        'node_basis': model.nodes_basis,
        'nodes': [model.nodes.shape, model.nodes.dtype],
        'number_of_elements': model.nb_elements,
        'elements': [model.elements.shape, model.elements.dtype],
        'materials': [model.materials.shape, model.materials.dtype]
    })

    
if __name__ == "__main__":
    app()