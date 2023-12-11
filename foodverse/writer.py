from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer
from typing import List, Optional, Dict
import queue
import numpy as np
import io
import json
import traceback
import os
import pickle

from foodverse.configs import sim_configs as sc
from foodverse.configs import usd_configs as uc
from foodverse.utils import dataset_utils as du
from foodverse.utils import file_utils as fu

__version__ = "0.0.1"

class FoodverseWriter(Writer):
    def __init__(
            self,
            output_dir: str,
            backend: BackendDispatch,
            render_products: List["og.Node"],
            semantic_types: List[str] = None,
            rgb: bool = False,
            bounding_box_2d_loose: bool = False,
            semantic_segmentation: bool = False,
            instance_id_segmentation: bool = False,
            instance_segmentation: bool = False,
            amodal_segmentation: bool = False,
            distance_to_camera: bool = False,
            normals: bool = False,
            bounding_box_3d: bool = False,
            occlusion: bool = False,
            camera_params: bool = False,
            frame_padding: int = 4,
            image_output_format: str = "png",
            scene_start_idx: int = 0,
            ) -> None:
        
        self._frame_id = scene_start_idx
        self.version = __version__
        self._output_dir = output_dir
        self.backend = backend
        self._frame_padding = frame_padding
        self._image_output_format = image_output_format

        self.render_products = render_products
        self.rp_labels = [rp.split("/")[-1] for rp in render_products]
        self.annotators = {}

        self.colorize_semantic_segmentation = False
        self.colorize_instance_id_segmentation = False
        self.colorize_instance_segmentation = False

        self.semantic_mapping = uc.SEMANTIC_MAPPINGS
        self.amodal_segmentation = amodal_segmentation
        self.scene_metadata = None

        # TODO(snair): Don't hardcode 105
        self.instance_colour_palette = np.array(
            du.generate_unique_colors(N=105), dtype=np.uint8)
        self.metadata_keys_to_write = ["class", "model_label",
                                        "semantic_id", "instance_id"]

        # Specify the semantic types that will be included in output
        if semantic_types is None:
            semantic_types = ["class"]

        # RGB
        if rgb:
            self.add_annotator("rgb")
            
        if bounding_box_2d_loose:
            self.add_annotator("bounding_box_2d_loose",
                               semantic_types=semantic_types)
            
        # Semantic Segmentation
        if semantic_segmentation:
            self.add_annotator("semantic_segmentation",
                               semantic_types=semantic_types,
                               colorize=self.colorize_semantic_segmentation)
            
        # Instance ID Segmentation
        # NOTE: Instance ID records prims regardless of whether or not they
        # have semantic labels. Hence, do not pass in semantic types.
        if instance_id_segmentation:
            self.add_annotator("instance_id_segmentation",
                               colorize=self.colorize_instance_id_segmentation)
            
        # Instance Segmentation
        if instance_segmentation:
            self.add_annotator("instance_segmentation",
                               semantic_types=semantic_types,
                               colorize=self.colorize_instance_segmentation)
            
        # Depth
        if distance_to_camera:
            self.add_annotator("distance_to_camera")

        # Normals
        if normals:
            self.add_annotator("normals")
        
        # Bounding Box 3D
        if bounding_box_3d:
            self.add_annotator("bounding_box_3d",
                               semantic_types=semantic_types)
            
        # Occlusion
        if occlusion:
            self.add_annotator("occlusion")

        # Camera Params
        if camera_params:
            self.add_annotator("camera_params")
        
    def fetch_data_from_annotator(self, annotator_name: str, data_dict: Dict):
        """Fetches data from the specified annotator.

        Args:
            annotator_name (str): Name of the annotator.
            data_dict (dict): Dictionary to append the data to.

        Returns:
            None. Fetched data is appended to the data_dict.
        """
        # Validate that the annotator has been registered.
        if annotator_name not in self.annotators:
            raise ValueError(
                f"Annotator {annotator_name} has not been registered, "
                f"cannot fetch data.")

        for rp_idx, annotator in enumerate(self.annotators[annotator_name]):
            rp_label = self.rp_labels[rp_idx]
            annotator_label = f"{annotator_name}-{rp_label}"
            data_dict[annotator_label] = annotator.get_data()
    
    def fetch_data(self) -> dict:
        """Fetches data from the registered annotators.

        Returns:
            A dictionary containing the data.
        """
        data = {}
        for annotator_name in self.annotators:
            if annotator_name == "camera_params":
                # Don't need to fetch camera params every scene since our
                # cameras are static.
                continue

            self.fetch_data_from_annotator(annotator_name, data)

        return data
    
    def write_data(self, prim_names_to_expect: List[str]) -> None:
        """Writes data to the output directory.

        Returns:
            None
        """

        try:
            data = self.fetch_data()
            self.write(data, prim_names_to_expect)
        except:
            traceback_msg = traceback.format_exc()
            self._handle_error("Error writing visible data",
                               traceback_msg, data, "visible")
            self._frame_id += 1
            raise

        self._frame_id += 1

    def _handle_error(self, message: str, traceback_msg: str,
                      data: Dict, stage: str) -> None:
        """Handles errors that occur during writing.

        Args:
            error_frame_id (int): The frame ID that the error occurred on.
            message (str): The error message.
            traceback (str): The traceback.
            stage (str): The stage of writing that the error occurred on.
                Should be one of "visible" or "amodal".
        Returns:
            None
        """
        padded_frame_id = f"{self._frame_id:0{self._frame_padding}}"
        scene_name = f"scene_{padded_frame_id}"
        filename = f"{padded_frame_id}_error_{stage}"
        file_basename = fu.compose_file_basename("errors", scene_name, filename)
        data_dump_path = fu.compose_file_path(file_basename, "pkl")
        error_output = {
            "frame_id": self._frame_id,
            "message": message,
            "traceback": traceback_msg,
            "dump_path": data_dump_path,
        }

        buf = io.BytesIO()
        pickle.dump(data, buf)
        buf.seek(0)

        self.backend.write_blob(data_dump_path, buf.getvalue())
        self._write_dict_to_json(error_output, file_basename)

    def write_amodal(self, expected_frame_id: int,
                     expected_prim_path: str,
                     instance_segmentation: bool = True,
                     normals: bool = False,
                     distance_to_camera: bool = False) -> None:
        """Writes amodal segmentation data for prim to the output directory.
        
        Args:
            expected_frame_id (int): The expected frame ID.
            expected_prim_path (str): The expected prim path.
        
        Returns:
            None
        """
        data = {}
        if instance_segmentation:
            self.fetch_data_from_annotator("instance_segmentation", data)

        if normals:
            self.fetch_data_from_annotator("normals", data)

        if distance_to_camera:
            self.fetch_data_from_annotator("distance_to_camera", data)

        try:
            self._process_amodal(expected_frame_id, expected_prim_path, data)
        except:
            traceback_msg = traceback.format_exc()
            self._handle_error("Error writing amodal data",
                               traceback_msg, data, "amodal")
            raise

    def _process_amodal(self, expected_frame_id: int,
                        expected_prim_path: str, data: Dict) -> None:
        """Processes amodal segmentation data for output.

        Args:
            expected_frame_id (int): The expected frame ID.
            expected_prim_path (str): The expected prim path.
            data (dict): The data to process.

        Returns:
            None
        """
        scene_name = self.scene_metadata["scene_name"]
        padded_frame_id = self.scene_metadata["padded_frame_id"]
        frame_id = self.scene_metadata["frame_id"]
        metadata = self.scene_metadata["metadata"]
        
        if frame_id != expected_frame_id:
            raise ValueError(f"Expected frame ID {expected_frame_id} but got "
                                f"{frame_id}")
        
        if expected_prim_path not in metadata:
            raise ValueError(f"Expected prim path {expected_prim_path} not "
                                f"found in metadata.")

        instance_id = metadata[expected_prim_path]["instance_id"]
        self._write_annotations(data, metadata, scene_name, padded_frame_id, instance_id)


    def add_annotator(self, annotator_name: str,
                      semantic_types: Optional[List[str]] = None,
                      colorize: Optional[bool] = None):
        
        init_params = {}
        if semantic_types is not None:
            init_params["semanticTypes"] = semantic_types
        
        if colorize is not None:
            init_params["colorize"] = colorize
        
        # If no init params were specified, set to None.
        if not init_params:
            init_params = None
        
        # Add annotator to dictionary.
        self.annotators[annotator_name] = []
        for rp in self.render_products:
            annotator = AnnotatorRegistry.get_annotator(
                            annotator_name, init_params=init_params)
            annotator.attach([rp])
            self.annotators[annotator_name].append(annotator)

    def write(self, data: Dict,
              prim_names_to_expect: Optional[List[str]] = None) -> None:
        """Write function called from the OgnWriter node on every frame to
        process annotator output.
        
        Args:
            data (dict): Dictionary containing the data to write.

        Returns:
            None
        """
        padded_frame_id = f"{self._frame_id:0{self._frame_padding}}"
        scene_name = f"scene_{padded_frame_id}"

        # Build metadata mapping so that all outputs can be uniquely identified
        # based on the instance_label.
        annotator_names = data.keys()
        
        metadata = self._build_metadata_mapping(data, annotator_names, prim_names_to_expect)
        self._store_metadata(metadata, scene_name, padded_frame_id)
        
        self._write_annotations(data, metadata, scene_name, padded_frame_id)

        filename = f"{padded_frame_id}_metadata"
        self._write_metadata(metadata, scene_name, filename)

    def _write_annotations(self, data: Dict, metadata: Dict, scene_name: str,
                           padded_frame_id: str, instance_id: Optional[int] = None) -> None:
        """Parses annotations from the data dictionary and dispatches them to
        the appropriate annotator writer.

        Args:
            data: Dictionary containing the data to write.
            metadata: Dictionary containing metadata.
            scene_name: Name of the scene.
            padded_frame_id: Padded frame ID.
            instance_id: Instance ID of the prim to write (only for amodal data).
                If specified, supported annotator writers will write data to a 
                directory prepended with `amodal`, and the filename will be
                suffixed with the instance ID.

        Returns:
            None
        """
        basename_prefix = None
        if instance_id is not None:
            # If instance ID is specified, then we are writing amodal data.
            # NOTE: Not all annotator writers support amodal data.
            basename_prefix = "amodal"

        # Annotator names are of the form 
        # "<annotator-name>-RenderProduct_Viewport_<Viewport Idx (optional)>"
        # E.g, "rgb-RenderProduct_Viewport_2" or "rgb-RenderProduct_Viewport"
        # Viewport index is ommitted for the first render product viewport.
        # If there is only one render product, then the annotator name only
        # consists of the annotator name (e.g "rgb" or "bounding_box_2d_loose")
        for annotator_name in data:            
            render_product_idx = du.get_idx_from_annotator_name(annotator_name)
            
            # We want each filename to use the following naming convention:
            # <frame_id>_viewport_<viewport_idx>.<ext>
            viewport_name = du.compose_viewport_name(render_product_idx)
            filename = f"{padded_frame_id}_{viewport_name}"

            if instance_id is not None:
                # If instance id is specified, then append it to the filename. 
                filename = f"{filename}_prim_{instance_id}"

            if annotator_name.startswith("rgb"):
                self._write_rgb(data, annotator_name, scene_name, filename)

            if annotator_name.startswith("semantic_segmentation"):
                self._write_semantic_segmentation(data, annotator_name,
                                                  scene_name, filename,
                                                  metadata, colorize=True)

            if annotator_name.startswith("instance_segmentation"):
                self._write_instance_segmentation(data, annotator_name,
                                                  scene_name, filename,
                                                  metadata, colorize=True,
                                                  basename_prefix=basename_prefix)

            if annotator_name.startswith("distance_to_camera"):
                self._write_depth_image(data, annotator_name,
                                        scene_name,filename,
                                        basename_prefix)

            if annotator_name.startswith("normals"):
                self._write_normals(data, annotator_name,
                                    scene_name, filename,
                                    basename_prefix)

            if annotator_name.startswith("bounding_box_3d"):
                self._write_bounding_box(data, "bounding_box_3d",
                                         annotator_name, scene_name,
                                         filename, metadata)

            if annotator_name.startswith("bounding_box_2d_loose"):
                self._write_bounding_box(data, "bounding_box_2d",
                                         annotator_name, scene_name,
                                         filename, metadata)

    def _write_rgb(self, data: Dict, annotator: str, scene_name: str,
                   filename: str) -> None:
        """Write RGB image to disk asynchronously.
        
        Args:
            data (dict): Dictionary containing the data to write.
            annotator (str): Annotator name.
            scene_name (str): Name of the scene.
            filename (str): Filename to write to.

        Returns:
            None
        """
        # Get RGB image.
        rgb_image = data[annotator]

        ext = self._image_output_format
        file_basename = fu.compose_file_basename("images", scene_name, filename)
        file_path = fu.compose_file_path(file_basename, ext)

        # Write RGB image to disk.
        self.backend.write_image(file_path, rgb_image)

    def _write_instance_segmentation(self, data: Dict, annotator: str,
                                     scene_name: str, filename: str,
                                     metadata: Dict, colorize: bool = True,
                                     basename_prefix: Optional[str] = None) -> None:
        """Write instance segmentation masks to disk asynchronously.

        Args:
            data: Dictionary containing the data to write.
            annotator: Annotator name.
            scene_name: Name of the scene.
            filename: Filename to write to.
            metadata: Metadata mapping.
            colorize: Whether to colorize the instance segmentation masks.
            basename_prefix (str): Prefix to use for the basename.

        Returns:
            None
        """
        # Get instance segmentation masks.
        instance_seg_data = data[annotator]["data"]
        height, width = instance_seg_data.shape[:2]

        ext = self._image_output_format
        file_basename = fu.compose_file_basename("instance_segmentation",
                                                 scene_name, filename, 
                                                 basename_prefix)
        file_path = fu.compose_file_path(file_basename, ext)

        # TODO(snair): Add check to make sure data is of the right shape.

        instance_seg_data = instance_seg_data.view(
            np.uint32).reshape(height, width)
        
        id_to_prim_paths = data[annotator]["info"]["idToLabels"]

        if colorize:
            # Stores a mapping from the instance ID output by instance_seg
            # annotator to remapped RGBA value.
            instance_id_from_replicator_to_rgba = {}
            # Stores a mapping from the remapped RGBA value to the true instance ID.
            rgba_to_true_instance_ids = {}
        else:
            # Stores a mapping from the instance ID from replicator to the true
            # instance ID.
            instance_id_to_true_id = {}

        for id in id_to_prim_paths:
            prim_path = id_to_prim_paths[id]
            try:
                prim_metadata = metadata[prim_path]
            except:
                assert False, "Error in getting prim metadata."
            instance_id_from_replicator = int(id)
            true_instance_id = prim_metadata["instance_id"]

            if colorize:
                rgba = self.instance_colour_palette[true_instance_id]
                instance_id_from_replicator_to_rgba[instance_id_from_replicator] = rgba
                rgba_to_true_instance_ids[tuple(rgba)] = true_instance_id
            else:
                instance_id_to_true_id[instance_id_from_replicator] = true_instance_id

        if colorize:
            remapped_instance_seg_data = du.id_to_rgba(
                semantic_array=instance_seg_data,
                semantic_id_to_rgba=instance_id_from_replicator_to_rgba
            )
            remapped_instance_seg_data = remapped_instance_seg_data.view(
                np.uint8).reshape(height, width, -1)
            self._write_dict_to_json(rgba_to_true_instance_ids, file_basename)
        else:
            remapped_instance_seg_data = du.remap_ids(
                semantic_array=instance_seg_data,
                semantic_id_to_true_id=instance_id_to_true_id
            )
        
        self.backend.write_image(file_path, remapped_instance_seg_data)

    def _write_bounding_box(self, data: Dict, bbox_type: str,
                                 annotator: str, scene_name: str,
                                 filename: str, metadata: Dict) -> None:
        """Write bounding box data to disk asynchronously.

        Args:
            data (dict): Dictionary containing the data to write.
            bbox_type (str): Type of bounding box.
            annotator (str): Annotator name.
            scene_name (str): Name of the scene.
            filename (str): Filename to write to.
            metadata (dict): Metadata mapping.

        Returns:
            None
        """
        # Get bounding box data.
        bbox_data = data[annotator]["data"]
        prim_paths = data[annotator]["info"]["primPaths"]
        SEMANTIC_ID_IDX = 0

        idx_to_instance_ids = {}

        for i, prim_path in enumerate(prim_paths):
            try:
                prim_metadata = metadata[prim_path]
            except:
                assert False, "Error in getting prim metadata."
            semantic_id = prim_metadata["semantic_id"]
            instance_id = prim_metadata["instance_id"]
            bbox_data[i][SEMANTIC_ID_IDX] = semantic_id

            idx_to_instance_ids[i] = instance_id

        ext = "npy"
        file_basename = fu.compose_file_basename(bbox_type,
                                                 scene_name, filename)
        file_path = fu.compose_file_path(file_basename, ext)
        buf = io.BytesIO()
        np.save(buf, bbox_data)
        self.backend.write_blob(file_path, buf.getvalue())

        self._write_dict_to_json(idx_to_instance_ids, file_basename)

    def write_camera_params(self) -> None:
        """Fetches and writes camera parameters to disk."""

        # Get camera parameters.
        data = {}
        self.fetch_data_from_annotator(
            annotator_name="camera_params", data_dict=data)

        for annotator_name in data:
            serializable_data = {}

            render_product_idx = du.get_idx_from_annotator_name(annotator_name)
            viewport_name = du.compose_viewport_name(render_product_idx)
            file_basename = f"camera_params/camera_params_{viewport_name}"

            camera_params = data[annotator_name]
            for key, val in camera_params.items():
                if isinstance(val, np.ndarray):
                    serializable_data[key] = val.tolist()
                else:
                    serializable_data[key] = val

            self._write_dict_to_json(serializable_data, file_basename)

    def _write_depth_image(self, data: Dict,
                           annotator: str,
                           scene_name: str,
                           filename: str,
                           basename_prefix: Optional[str] = None) -> None:
        """Write depth image to disk asynchronously.

        Args:
            data (dict): Dictionary containing the data to write.
            annotator (str): Annotator name.
            scene_name (str): Name of the scene.
            filename (str): Filename to write to.
            basename_prefix: Prefix to use for the basename.

        Returns:
            None
        """
        # Get depth image.
        depth_image = data[annotator]

        ext = "npy"
        file_basename = fu.compose_file_basename("depth_images",
                                                 scene_name, filename,
                                                 basename_prefix)
        file_path = fu.compose_file_path(file_basename, ext)

        buf = io.BytesIO()
        np.save(buf, depth_image)

        # Write depth image to disk.
        self.backend.write_blob(file_path, buf.getvalue())

    def _write_normals(self, data: Dict, annotator: str,
                       scene_name: str,
                       filename: str,
                       basename_prefix: Optional[str] = None) -> None:
        """Write normals to disk asynchronously.

        Args:
            data (dict): Dictionary containing the data to write.
            annotator (str): Annotator name.
            scene_name (str): Name of the scene.
            filename (str): Filename to write to.
            basename_prefix: Prefix to use for the basename.

        Returns:
            None
        """
        # Get normals.
        normals_data = data[annotator]

        ext = self._image_output_format
        file_basename = fu.compose_file_basename("normals",
                                                 scene_name, filename, 
                                                 basename_prefix)
        file_path = fu.compose_file_path(file_basename, ext)

        colorized_normals_data = du.colorize_normals(normals_data)
        self.backend.write_image(file_path, colorized_normals_data)


    def _write_semantic_segmentation(self, data: Dict, annotator: str,
                                        scene_name: str, filename: str,
                                        metadata: Dict, colorize: bool = True) -> None:
        """Write semantic segmentation masks to disk asynchronously.

        Args:
            data (dict): Dictionary containing the data to write.
            annotator (str): Annotator name.
            scene_name (str): Name of the scene.
            filename (str): Filename to write to.
            metadata (dict): Metadata mapping.

        Returns:
            None
        """
        # Get semantic segmentation masks.
        semantic_seg_data = data[annotator]["data"]
        height, width = semantic_seg_data.shape[:2]

        ext = self._image_output_format
        file_basename = fu.compose_file_basename("semantic_segmentation",
                                                 scene_name, filename)
        file_path = fu.compose_file_path(file_basename, ext)

        if self.colorize_semantic_segmentation:
            # Replicator annotator should not colorize the masks. Instead, the
            # pixels should contain the semantic id so that we can remap it
            # to a color from a custom palette.
            raise ValueError(
                "Writer does not accept colorized semantic segmentation masks.")

        semantic_seg_data = semantic_seg_data.view(
            np.uint32).reshape(height, width)
        id_to_labels = data[annotator]["info"]["idToLabels"]

        if colorize:
            # Stores a mapping from the semantic ID output by replicator to
            # remapped RGBA value.
            semantic_id_from_replicator_to_rgba = {}

            # Stores a mapping from the remapped RGBA value to the true semantic ID
            # (predefined for each class in the dataset).
            rgba_to_class_names = {}
        else:
            # Stores a mapping from the semantic ID from replicator to the true
            # semantic ID (predefined for each class in the dataset).
            semantic_id_to_true_id = {}
        for semantic_id_from_replicator, label in id_to_labels.items():
            model_label = label["class"]
            semantic_cfg = self.semantic_mapping[model_label]

            if colorize:
                semantic_id_from_replicator_to_rgba[
                    int(semantic_id_from_replicator)] = semantic_cfg.rgba

                rgba_to_class_names[
                    tuple(semantic_cfg.rgba)] = semantic_cfg.semantic_id
            else:   
                semantic_id_to_true_id[
                    int(semantic_id_from_replicator)] = semantic_cfg.semantic_id

        if colorize:
            remapped_semantic_seg_data = du.id_to_rgba(
                semantic_array=semantic_seg_data,
                semantic_id_to_rgba=semantic_id_from_replicator_to_rgba
            )
            remapped_semantic_seg_data = remapped_semantic_seg_data.view(
                np.uint8).reshape(height, width, -1)
            self._write_dict_to_json(rgba_to_class_names, file_basename)
        else:
            remapped_semantic_seg_data = du.remap_ids(
                semantic_array=semantic_seg_data,
                semantic_id_to_true_id=semantic_id_to_true_id
            )
        self.backend.write_image(file_path, remapped_semantic_seg_data)

    def _write_dict_to_json(self, output_dict: Dict,
                            file_basename: str) -> None:
        """Write a dictionary to the backend.
        
        Args:
            output_dict (dict): Dictionary to write.
            file_basename (str): Relative file path minus extension.
                Note that absolute path is determined by the backend.

        Returns:
            None
        """
        fu.write_dict_to_json_backend(output_dict, self.backend, file_basename)

    def _trim_metadata(self, metadata: Dict) -> dict:
        """Trim metadata to only contain the keys that are required.

        Args:
            metadata (dict): Metadata mapping.

        Returns:
            dict: Trimmed metadata mapping.
        """
        output_metadata = {}

        for prim_path, prim_metadata in metadata.items():
            trimmed_metadata = {}

            if prim_metadata["model_label"] == "scene":
                continue

            for key in prim_metadata:
                if key in self.metadata_keys_to_write:
                    trimmed_metadata[key] = prim_metadata[key]

            _, prim_name = du.decompose_prim_path(prim_path)
            output_metadata[prim_name] = trimmed_metadata

        return output_metadata

    def _store_metadata(self, metadata: Dict, scene_name: str,
                          padded_frame_id: str) -> None:
        """Saves the metadata for the current scene so that we can use it
        later on while processing the amodal segmentation masks.

        Args:
            metadata (dict): Metadata mapping.
            scene_name (str): Name of the scene.
            padded_frame_id (str): Padded frame ID.

        Returns:
            None
        """

        self.scene_metadata = {
            "scene_name": scene_name,
            "padded_frame_id": padded_frame_id,
            "frame_id": self._frame_id,
            "metadata": metadata
        }

    def _write_metadata(self, metadata: Dict,
                        scene_name: str, file_name: str) -> None:
        """Write metadata to disk asynchronously.

        Args:
            metadata (dict): Metadata mapping.
            scene_name (str): Name of the scene.
            file_name (str): Filename to write to.

        Returns:
            None
        """
        # Trim metadata to only contain the keys that are required.
        output_metadata = self._trim_metadata(metadata)

        file_basename = f"metadata/{scene_name}/{file_name}"

        self._write_dict_to_json(output_metadata, file_basename)

    def _build_metadata_mapping(self, data: Dict,
                                annotator_names: List[str],
                                prim_names_to_expect: List[str]) -> dict:
        """Builds a metadata mapping from the data dictionary.
        
        Args:
            data (dict): Dictionary containing the data to write.
            annotator_names (List[str]): List of annotator names.
            prim_names_to_expect (List[str]): List of mesh names to expect.
                If we receive an unexpected prim in the data, we will raise an
                error.

        Raises:
            ValueError: If we receive an unexpected prim in the data.

        Returns:
            dict: Metadata mapping of the form:
                {
                    INSTANCE_PRIM_PATH_1: {
                        "class": "salad_chicken_strip",
                        "model_label": "id_3_salad_chicken_strip_10g",
                        "semantic_id": X,
                        "instance_id": Y,
                        "id_from_instance_seg_annotator": Z,
                        "id_from_instance_id_seg_annotator": W,
                    },
                    INSTANCE_PRIM_PATH_2: {...},
                }
            NOTE: The instance label has form "id_3_salad_chicken_strip_10g_1"
        """
        metadata = {}
        bbox_3d_loose_annotator_names = list(filter(
            lambda annotator_name: annotator_name.startswith(
            "bounding_box_3d"), annotator_names))
        bbox_3d_loose_annotator_name = bbox_3d_loose_annotator_names[0]
        bbox_3d_loose = data[bbox_3d_loose_annotator_name]
        prim_paths_bbox_3d_loose = bbox_3d_loose["info"]["primPaths"]

        SCENE_INSTANCE_ID = 0
        PLATE_INSTANCE_ID = 1
        model_label = "UNLABELLED"
        semantic_cfg = self.semantic_mapping[model_label]
        metadata[model_label] = {
                "class": semantic_cfg.class_name,
                "model_label": model_label,
                "semantic_id": semantic_cfg.semantic_id,
                "instance_id": SCENE_INSTANCE_ID,
            }

        food_instance_id_counter = 2

        expected_prims_set = set(prim_names_to_expect)
        expected_prims_set.add("scene")
        expected_prims_set.add("plate")

        received_prims_set = set()
        for prim_path in prim_paths_bbox_3d_loose:
            if prim_path == '/Replicator/Ref_Xform/Ref':
                model_label = "scene"
                prim_name = "scene"
                instance_id = SCENE_INSTANCE_ID
            else:
                _, prim_name = du.decompose_prim_path(prim_path)
                model_label, _ = du.decompose_prim_name(prim_name)
                if model_label == "plate":
                    instance_id = PLATE_INSTANCE_ID
                else:
                    instance_id = food_instance_id_counter
                    food_instance_id_counter += 1
            
            # Get class name and semantic id from model label.
            semantic_cfg = self.semantic_mapping[model_label]
            metadata[prim_path] = {
                "class": semantic_cfg.class_name,
                "model_label": model_label,
                "semantic_id": semantic_cfg.semantic_id,
                "instance_id": instance_id,
            }
            received_prims_set.add(prim_name)

        if received_prims_set != expected_prims_set:
            raise ValueError(
                f"Received unexpected prims: {received_prims_set - expected_prims_set}")

        return metadata
