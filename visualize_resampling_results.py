#!/usr/bin/env python3
"""
Simple visualization: TIC image and average mass spectrum only.
"""

import matplotlib.pyplot as plt
import numpy as np
import spatialdata as sd


def visualize_resampling_results(zarr_path):
    """Visualize TIC image and average mass spectrum."""

    # Load the SpatialData
    sdata = sd.read_zarr(zarr_path)
    table_name = list(sdata.tables.keys())[0]
    table = sdata.tables[table_name]

    # Get data
    mz_values = table.var["mz"].values
    if hasattr(table.X, "toarray"):
        X_dense = table.X.toarray()
    else:
        X_dense = table.X

    # Calculate average spectrum and TIC per pixel
    avg_spectrum = X_dense.mean(axis=0)
    tic_per_pixel = X_dense.sum(axis=1)

    # Get spatial coordinates
    coords = table.obs[["x", "y"]].values
    x_coords, y_coords = coords[:, 0], coords[:, 1]

    # Reshape TIC into 2D image
    x_unique = np.unique(x_coords)
    y_unique = np.unique(y_coords)
    tic_image = np.zeros((len(y_unique), len(x_unique)))

    for i, tic_val in enumerate(tic_per_pixel):
        x_idx = np.where(x_unique == x_coords[i])[0][0]
        y_idx = np.where(y_unique == y_coords[i])[0][0]
        tic_image[y_idx, x_idx] = tic_val

    # Create simple 2-panel plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Average Mass Spectrum
    ax1.plot(mz_values, avg_spectrum, "b-", linewidth=0.8)
    ax1.set_title("Average Mass Spectrum")
    ax1.set_xlabel("m/z")
    ax1.set_ylabel("Average Intensity")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Total Ion Count Image
    im = ax2.imshow(tic_image, cmap="viridis", aspect="equal")
    ax2.set_title("Total Ion Count Image")
    ax2.set_xlabel("X Position")
    ax2.set_ylabel("Y Position")
    plt.colorbar(im, ax=ax2, label="TIC")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_resampling_results(
        r"C:\Users\P70078823\Desktop\MSIConverter\output_zarr_bruker.zarr"
    )
