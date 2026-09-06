"""
Shared File & Storage Utilities for Forensic Layer
"""

import os
import json
from pathlib import Path


def get_image_files(directory) -> list[str]:
    """
    Returns a sorted list of image filenames (.jpg, .jpeg, .png) in the given directory.

    Parameters:
        directory (str | Path): Path to directory.

    Returns:
        list[str]: Sorted list of matching filenames (not full paths).
    """
    if not os.path.exists(directory):
        return []
    return sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])


def save_embedding(document_id: str, data: dict, layer_slug: str, results_dir=None) -> Path:
    """
    Saves a layer measurement / feature embedding dictionary to JSON at:
    results/embeddings/{document_id}_{layer_slug}.json

    Parameters:
        document_id (str): Identifier of the document (e.g. filename stem).
        data (dict): Measurement dictionary to serialize.
        layer_slug (str): Layer identifier slug (e.g. 'layer2_photo_boundary').
        results_dir (str | Path, optional): Custom results directory. Defaults to <project_root>/results.

    Returns:
        Path: Path to the written embedding file.
    """
    if results_dir is None:
        results_dir = Path(__file__).resolve().parent.parent / 'results'
    embeddings_dir = Path(results_dir) / 'embeddings'
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    out_path = embeddings_dir / f"{document_id}_{layer_slug}.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)
    return out_path
