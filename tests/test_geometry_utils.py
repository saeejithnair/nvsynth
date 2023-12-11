import unittest
import numpy as np
from foodverse.utils.geometry_utils import *

# Replace pxr.Usd types with numpy arrays for mocking purposes
class MockGfRange3d:
    def __init__(self, min_val, max_val):
        self._min = min_val
        self._max = max_val

    def GetMin(self):
        return self._min

    def GetMax(self):
        return self._max

    def GetSize(self):
        return self._max - self._min

class MockGfBBox3d:
    def __init__(self, min_val, max_val):
        self._range = MockGfRange3d(min_val, max_val)

    def GetRange(self):
        return self._range

    def ComputeCentroid(self):
        return (self._range.GetMin() + self._range.GetMax()) / 2

class TestGeometryUtils(unittest.TestCase):
    def test_bbox(self):
        min_point = np.array([0, 0, 0])
        max_point = np.array([2, 4, 6])
        bbox = MockGfBBox3d(min_point, max_point)
        wrapped_bbox = BBox(bbox)

        self.assertTrue(np.all(wrapped_bbox.min == min_point))
        self.assertTrue(np.all(wrapped_bbox.max == max_point))
        self.assertEqual(wrapped_bbox.x_dim, 2)
        self.assertEqual(wrapped_bbox.y_dim, 4)
        self.assertEqual(wrapped_bbox.z_dim, 6)
        self.assertEqual(wrapped_bbox.largest_dim, 6)
        self.assertEqual(wrapped_bbox.radius, 1)

    def test_point_within_circle(self):
        radius = 2
        point_inside = np.array([1, 1, 0])
        point_outside = np.array([3, 3, 0])

        self.assertTrue(point_within_circle(radius, point_inside))
        self.assertFalse(point_within_circle(radius, point_outside))

        point_on_edge = np.array([2, 0, 0])
        self.assertTrue(point_within_circle(radius, point_on_edge))

        point_outside_edge = np.array([2.1, 0, 0])
        self.assertFalse(point_within_circle(radius, point_outside_edge))

        self.assertTrue(point_within_circle(radius, point_outside_edge, radius_scale=1.1))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
