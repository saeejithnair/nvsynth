import unittest
from foodverse.utils import dataset_utils as du
import numpy as np
class TestDecomposePrimName(unittest.TestCase):

    def test_non_food_prim(self):
        input_str = "plate"
        expected_output = ("plate", None)
        actual_output = du.decompose_prim_name(input_str)
        self.assertEqual(expected_output, actual_output)

    def test_food_prim(self):
        input_str = "id_94_chicken_wing_27g_1"
        expected_output = ("id_94_chicken_wing_27g", 1)
        actual_output = du.decompose_prim_name(input_str)
        self.assertEqual(expected_output, actual_output)

    def test_food_prim_large_id(self):
        input_str = "id_94_steak_27g_9999"
        expected_output = ("id_94_steak_27g", 9999)
        actual_output = du.decompose_prim_name(input_str)
        self.assertEqual(expected_output, actual_output)

class TestDecomposePrimPath(unittest.TestCase):
    def test_non_food_prim(self):
        input_str = "/MyScope/plate"
        expected_output = ("/MyScope", "plate")
        actual_output = du.decompose_prim_path(input_str)
        self.assertEqual(expected_output, actual_output)

    def test_food_prim(self):
        input_str = "/MyScope/id_94_chicken_wing_27g_1"
        expected_output = ("/MyScope", "id_94_chicken_wing_27g_1")
        actual_output = du.decompose_prim_path(input_str)
        self.assertEqual(expected_output, actual_output)

class TestComposePrimName(unittest.TestCase):

    def test_without_id(self):
        model_label = "plate"
        expected_output = "plate"
        actual_output = du.compose_prim_name(model_label)
        self.assertEqual(expected_output, actual_output)

    def test_with_id(self):
        model_label = "id_94_chicken_wing_27g"
        id = 1
        expected_output = "id_94_chicken_wing_27g_1"
        actual_output = du.compose_prim_name(model_label, id)
        self.assertEqual(expected_output, actual_output)

    def test_food_model_without_id(self):
        model_label = "id_94_chicken_wing_27g"
        with self.assertRaises(ValueError):
            du.compose_prim_name(model_label)

    def test_non_food_model_with_id(self):
        model_label = "non_food_model"
        id = 2
        expected_output = "non_food_model_2"
        actual_output = du.compose_prim_name(model_label, id)
        self.assertEqual(expected_output, actual_output)


class TestMapFoodModelLabelToClassName(unittest.TestCase):
    def test_valid_model_labels(self):
        test_cases = [
            ("id_94_chicken_wing_27g", "chicken_wing"),
            ("id_1_salad_chicken_strip_7g", "salad_chicken_strip"),
            ("id_123_pizza_slice_100g", "pizza_slice"),
            ("id_37_costco_cucumber_sushi_roll_1_16g", "costco_cucumber_sushi_roll"),
            ("id_46_costco_salad_sushi_roll_30000_29000g", "costco_salad_sushi_roll"),
            ("id_5100009_costco_shrimp_sushi_roll_40_28g", "costco_shrimp_sushi_roll"),
        ]

        for model_label, expected_class_name in test_cases:
            self.assertEqual(
                du.map_food_model_label_to_class_name(model_label),
                expected_class_name)

    def test_invalid_model_labels(self):
        test_cases = [
            "chicken_wing_27g",
            "id_chicken_wing_27g",
            "id_94_chicken_wing",
            "id_94_chicken_wing_27",
        ]

        for model_label in test_cases:
            with self.assertRaises(ValueError):
                du.map_food_model_label_to_class_name(model_label)

class TestIDToRGBA(unittest.TestCase):

    def test_id_to_rgba(self):
        semantic_array = np.array([
            [0, 1, 2],
            [2, 1, 0]
        ])

        semantic_id_to_rgba = {
            0: [255, 0, 0, 255],
            1: [0, 255, 0, 255],
            2: [0, 0, 255, 255]
        }

        expected_rgba_array = np.array([
            [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]],
            [[0, 0, 255, 255], [0, 255, 0, 255], [255, 0, 0, 255]]
        ], dtype=np.uint8)

        result_rgba_array = du.id_to_rgba(semantic_array, semantic_id_to_rgba)
        np.testing.assert_array_equal(result_rgba_array, expected_rgba_array)

    def test_id_to_rgba_non_consecutive(self):
        semantic_array = np.array([
            [7, 1, 2],
            [2, 5, 0]
        ])

        semantic_id_to_rgba = {
            0: [255, 0, 0, 255],
            5: [5, 10, 15, 255],
            1: [0, 255, 0, 255],
            7: [7, 14, 21, 255],
            2: [0, 0, 255, 255],
        }

        for id in semantic_id_to_rgba:
            semantic_id_to_rgba[id] = np.array(
                semantic_id_to_rgba[id], dtype=np.uint8)

        expected_rgba_array = np.array([
            [[7, 14, 21, 255], [0, 255, 0, 255], [0, 0, 255, 255]],
            [[0, 0, 255, 255], [5, 10, 15, 255], [255, 0, 0, 255]]
        ], dtype=np.uint8)

        result_rgba_array = du.id_to_rgba(semantic_array, semantic_id_to_rgba)
        np.testing.assert_array_equal(result_rgba_array, expected_rgba_array)

    def test_id_to_rgba_non_consecutive_offset_by_1(self):
        semantic_array = np.array([
            [8, 1, 2],
            [2, 5, 7]
        ])

        semantic_id_to_rgba = {
            8: [255, 0, 0, 255],
            5: [5, 10, 15, 255],
            1: [0, 255, 0, 255],
            7: [7, 14, 21, 255],
            2: [0, 0, 255, 255],
        }
        for id in semantic_id_to_rgba:
            semantic_id_to_rgba[id] = np.array(
                semantic_id_to_rgba[id], dtype=np.uint8)

        expected_rgba_array = np.array([
            [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]],
            [[0, 0, 255, 255], [5, 10, 15, 255], [7, 14, 21, 255]]
        ], dtype=np.uint8)

        result_rgba_array = du.id_to_rgba(semantic_array, semantic_id_to_rgba)
        np.testing.assert_array_equal(result_rgba_array, expected_rgba_array)

    def test_id_to_rgba_non_consecutive_offset_by_2(self):
        semantic_array = np.array([
            [8, 15, 2],
            [2, 5, 7]
        ])

        semantic_id_to_rgba = {
            8: [255, 0, 0, 255],
            5: [5, 10, 15, 255],
            15: [0, 255, 0, 255],
            7: [7, 14, 21, 255],
            2: [0, 0, 255, 255],
        }
        for id in semantic_id_to_rgba:
            semantic_id_to_rgba[id] = np.array(
                semantic_id_to_rgba[id], dtype=np.uint8)

        expected_rgba_array = np.array([
            [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]],
            [[0, 0, 255, 255], [5, 10, 15, 255], [7, 14, 21, 255]]
        ], dtype=np.uint8)

        result_rgba_array = du.id_to_rgba(semantic_array, semantic_id_to_rgba)
        np.testing.assert_array_equal(result_rgba_array, expected_rgba_array)

class TestGetPrimPathFromInstanceLabel(unittest.TestCase):

    def test_food(self):
        instance_label = '/MyScope/id_42_costco_california_sushi_roll_2_27g_4/poly/mesh'
        expected_result = '/MyScope/id_42_costco_california_sushi_roll_2_27g_4'
        self.assertEqual(du.get_prim_path_from_instance_label(
            instance_label), expected_result)

    def test_food_textured(self):
        instance_label = '/MyScope/id_2_salad_chicken_strip_9g_7/textured/mesh'
        expected_result = '/MyScope/id_2_salad_chicken_strip_9g_7'
        self.assertEqual(du.get_prim_path_from_instance_label(
            instance_label), expected_result)

    def test_table(self):
        instance_label = '/Replicator/Ref_Xform/Ref/table_low_327/table_low'
        expected_result = '/Replicator/Ref_Xform/Ref'
        self.assertEqual(du.get_prim_path_from_instance_label(
            instance_label), expected_result)

    def test_plate(self):
        instance_label = '/MyScope/plate/model/mesh'
        expected_result = '/MyScope/plate'
        self.assertEqual(du.get_prim_path_from_instance_label(
            instance_label), expected_result)

class TestGenerateUniqueColors(unittest.TestCase):
    def test_basic(self):
        N = 5
        colors = du.generate_unique_colors(N)
        self.assertEqual(len(colors), N)
        self.assertFalse(du.has_duplicates([tuple(color) for color in colors]))

    def test_large_N(self):
        N = 100
        colors = du.generate_unique_colors(N)
        self.assertEqual(len(colors), N)
        self.assertFalse(du.has_duplicates([tuple(color) for color in colors]))

    def test_no_alpha(self):
        N = 10
        colors = du.generate_unique_colors(N)
        for color in colors:
            self.assertEqual(color[-1], 255)

    def test_correct_rgb_range(self):
        N = 50
        colors = du.generate_unique_colors(N)
        for color in colors:
            for channel in color[:3]:
                self.assertTrue(0 <= channel <= 255)

if __name__ == '__main__':
    unittest.main()
