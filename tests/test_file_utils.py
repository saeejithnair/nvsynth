import unittest
import os
import shutil
import tempfile

from foodverse.utils.file_utils import create_new_folder, remove_folder_if_exists

class TestFileOperations(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_create_new_folder(self):
        folder_name = "test_folder"
        new_folder_path = create_new_folder(self.temp_dir, folder_name)
        self.assertTrue(os.path.exists(new_folder_path))
        self.assertTrue(os.path.isdir(new_folder_path))

    def test_remove_folder_if_exists(self):
        folder_name = "test_folder"
        folder_path = os.path.join(self.temp_dir, folder_name)
        os.makedirs(folder_path)
        self.assertTrue(os.path.exists(folder_path))
        remove_folder_if_exists(folder_path)
        self.assertFalse(os.path.exists(folder_path))

if __name__ == '__main__':
    unittest.main()
