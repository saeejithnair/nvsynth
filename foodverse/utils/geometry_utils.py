"""Utility functions for geometry operations."""

import numpy as np

class BBox:
    """Wrapper around pxr.Usd.GfBBox3d for convenience."""
    def __init__(self, bbox: "pxr.Usd.GfBBox3d"):
        self._bbox = bbox
        self._range = bbox.GetRange()
        self._size = self._range.GetSize()

        # Radius of the bounding circle enclosing the projection of the
        # bounding box onto the XY plane.
        self._radius = min(self._size[0], self._size[1]) / 2.0
    
    @property
    def min(self):
        """Alias for minimum point of bounding box."""
        return self._range.GetMin()
    
    @property
    def max(self):
        """Alias for maximum point of bounding box."""
        return self._range.GetMax()
    
    @property
    def centroid(self):
        """Alias for centroid of bounding box."""
        return self._bbox.ComputeCentroid()
    
    @property
    def x_dim(self) -> float:
        """Alias for length of bounding box along x-dimension."""
        return self._size[0]
    
    @property
    def y_dim(self) -> float:
        """Alias for length of bounding box along y-dimension."""
        return self._size[1]
        
    @property
    def z_dim(self) -> float:
        """Alias for length of bounding box along z-dimension."""
        return self._size[2]
    
    @property
    def largest_dim(self):
        """Alias for largest dimension of bounding box."""
        return max(self._size[0], self._size[1], self._size[2])
    
    @property
    def radius(self):
        """Alias for radius of circle enclosing the projection of the
        bounding box onto the XY plane. Calculated as min(x_dim, y_dim) / 2.0.
        """
        return self._radius

def point_within_circle(radius: float, point: "pxr.Gf.GfVec3d",
                        radius_scale: float = 1.0) -> bool:
    """Checks if the point is within the circle, given some error (scale)
    bounds. Assumes that the circle is centered at the origin.

    Args:
        radius: Radius of the circle.
        point: Point to check in 3D space [x, y, z].
        radius_scale: Scale factor for the radius to account for error.

    Returns:
        True if the point is within the circle, False otherwise.
    """
    point_magnitude_2d = np.sqrt(point[0]**2 + point[1]**2)

    return point_magnitude_2d <= (radius * radius_scale)

def fibonacci_sphere(n_points, radius):
    if n_points <= 1:
        return [np.array([0, 0, radius])]
    
    points = []
    increment = np.pi * (3 - np.sqrt(5))

    for i in range(1, n_points + 1):  # Start with i=1 to avoid points on the equator
        phi = i * increment
        cos_theta = 1 - 2 * i / (n_points + 1)
        sin_theta = np.sqrt(1 - cos_theta**2)

        x = radius * sin_theta * np.cos(phi)
        y = radius * sin_theta * np.sin(phi)
        z = abs(radius * cos_theta)

        points.append(np.array([x, y, z]))

    return points

def make_views(init_pos, num_cameras, radius):
    init_pos = np.array(init_pos)
    pts = fibonacci_sphere(num_cameras, radius)

    # Shift hemisphere points to the initial camera position
    pts = [pt + init_pos for pt in pts]

    return pts