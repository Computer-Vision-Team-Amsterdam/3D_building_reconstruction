"""
Computes which building vertices are visible from the panorama location (the
"observer").

This updated version uses the `shapely`/`numpy` ray-casting version
of the module, it does not need `scikit-geometry` (`skgeom`). That package has
no builds for Python >3.10, therefore it cannot be installed in this
environment. The function names and signatures are unchanged, `run.py` should
work as before
"""

import numpy as np
from shapely.geometry import Point, Polygon

TOL = 1e-4  # metres, tolerance for "vertex lies on the visibility boundary"


# Build the set of walls and building edges: replaces the `skgeom` arrangement with a plain `(n, 4)` array of segments `(x1, y1, x2, y2)`, including a square bounding box around the observer
def generate_arrangement(observer, building_polygons):
    """Collect all wall + building segments (replaces the skgeom arrangement)"""
    set_range = 60
    lx, ly = observer[0] - set_range, observer[1] - set_range
    rx, ry = observer[0] + set_range, observer[1] + set_range

    segments = [
        (lx, ly, lx, ry),
        (lx, ry, rx, ry),
        (rx, ry, rx, ly),
        (rx, ly, lx, ly),
    ]
    for poly in building_polygons:
        for i in range(1, len(poly)):
            segments.append((poly[i - 1][0], poly[i - 1][1], poly[i][0], poly[i][1]))
    return np.array(segments, dtype=float)


# Visibility polygon: casts a ray from the observer towards every segment endpoint, plus a small offset to either side, and keeps the nearest hit for each ray. The hit points form the visibility polygon
def get_visibility_polygon(arr, observer):
    """Visibility polygon from the observer via ray casting against all
    segments"""
    ox, oy = observer[0], observer[1]
    p = arr[:, 0:2]  # segment starts (n,2)
    s = arr[:, 2:4] - arr[:, 0:2]  # segment vectors (n,2)

    # Cast a ray at every segment endpoint, plus a tiny offset either side
    ends = np.vstack([arr[:, 0:2], arr[:, 2:4]])
    base = np.arctan2(ends[:, 1] - oy, ends[:, 0] - ox)
    eps = 1e-7
    angles = np.unique(np.concatenate([base - eps, base, base + eps]))

    d = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (m,2)
    po = p - np.array([ox, oy])  # (n,2)

    denom = d[:, None, 0] * s[None, :, 1] - d[:, None, 1] * s[None, :, 0]  # (m,n)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (po[None, :, 0] * s[None, :, 1] - po[None, :, 1] * s[None, :, 0]) / denom
        u = (po[None, :, 0] * d[:, None, 1] - po[None, :, 1] * d[:, None, 0]) / denom
    valid = (np.abs(denom) > 1e-12) & (t >= 0) & (u >= -1e-9) & (u <= 1 + 1e-9)
    t = np.where(valid, t, np.inf)
    t_min = t.min(axis=1)

    hit = np.isfinite(t_min)
    pts = np.stack([ox + t_min[hit] * d[hit, 0], oy + t_min[hit] * d[hit, 1]], axis=1)

    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def draw_arrangement(arr, vx, q, save_file=False):
    """Debug plot: walls/buildings in blue, visibility polygon in red"""
    from matplotlib import pyplot as plt

    plt.figure(figsize=(10, 10))
    for x1, y1, x2, y2 in arr:
        plt.plot([x1, x2], [y1, y2], "b")
    xs, ys = vx.exterior.xy
    plt.plot(xs, ys, "r:")
    plt.scatter(q[0], q[1], color="b")
    plt.savefig("vispol.png") if save_file else plt.show()


# Direction of the observer
def direction_of_observer(segment, observer):
    """
    Determine direction of point from line segment

    Code is inspired by geeksforgeeks.org/direction-point-line-segment/
    """

    # A at the left side and B at the right, sort list on y-axis.
    segment = sorted(segment, key=lambda k: [k[1], k[0]])

    # Select two points from the sorted list
    aX = segment[0][0]
    aY = segment[0][1]
    bX = segment[1][0]
    bY = segment[1][1]

    # Subtracting co-ordinates of point A from B and P, to make A as origin
    bX = bX - aX
    bY = bY - aY
    pX = observer[0] - aX
    pY = observer[1] - aY

    # Determine the cross product
    cross_product = (bX * pY) - (bY * pX)

    return cross_product


# Which building segments are visible
def segments_in_visibility_polygon(building_poly, visibility_polygon, observer):
    """Keep building vertices that are inside or on the visibility polygon"""
    segments = []

    # Loop over segments
    for idx_start in range(len(building_poly) - 1):
        segment = building_poly[idx_start : idx_start + 2]

        # Check if both points of the segment are visible
        if (
            visibility_polygon.distance(Point(segment[0][0], segment[0][1])) <= TOL
        ) and (visibility_polygon.distance(Point(segment[1][0], segment[1][1])) <= TOL):
            # Determine the direction of the observer from the line segment
            # Lies the observer to the Right of Line Segment or to the Left of Line Segment.
            cross_product = direction_of_observer(segment, observer)

            # Observer is on the right of line segment if cross product is positive
            if cross_product > 0:
                # Reverse the two points
                # segment = [segment[1], segment[0]]
                pass  # TODO looks like this is not needed, requires further testing

            segments.append(segment)

    return segments
