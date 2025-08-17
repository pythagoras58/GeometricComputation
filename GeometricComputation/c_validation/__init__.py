import numpy as np
from scipy.spatial import ConvexHull
import pandas as pd

def compare_hulls(n):
    # Load C-generated data
    points = pd.read_csv(f"points_{n}.csv", header=None).values
    c_hull = pd.read_csv(f"hull_{n}.csv", header=None).values

    # Compute SciPy's hull
    scipy_hull = ConvexHull(points)
    scipy_vertices = points[scipy_hull.vertices]

    # Normalize: sort vertices to handle different orderings
    def normalize(hull):
        # Find point with smallest x (then y) to standardize starting point
        min_idx = np.lexsort((hull[:,1], hull[:,0]))[0]
        return np.roll(hull, -min_idx, axis=0)  # Rotate to start with min point

    c_norm = normalize(c_hull)
    scipy_norm = normalize(scipy_vertices)

    # Check if hulls match (within floating point tolerance)
    match = np.allclose(c_norm, scipy_norm, atol=1e-6)
    print(f"Comparison for {n} points: {'MATCH' if match else 'MISMATCH'}")
    if not match:
        print(f"  C hull size: {len(c_hull)}, SciPy hull size: {len(scipy_vertices)}")

# Validate with test sizes
for n in [10000, 100000, 1000000]:
    compare_hulls(n)