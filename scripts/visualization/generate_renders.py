import bpy
import os
import math
import tyro
import re

# Generates GIFs from OBJ files using Blender for Nutritionverse 2.0 dataset
# Function to clear all objects from the scene
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

# Function to set up the camera
def setup_camera():
    bpy.ops.object.camera_add(location=(5, -5, 5))
    camera = bpy.context.object
    bpy.context.scene.camera = camera

# Function to set up lighting
def setup_lighting():
    bpy.ops.object.light_add(type='SUN', location=(10, -10, 10))
    light = bpy.context.object
    light.data.energy = 5

# Function to import and set up an object
def import_object(obj_path):
    # Clear the scene
    clear_scene()
    
    # Import the OBJ file
    bpy.ops.wm.obj_import(filepath=obj_path)
    
    # Center and scale the object
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')
            bpy.ops.object.shade_smooth()
            
            # Scale the object to fit the camera view
            scale_factor = 2 / max(obj.dimensions)
            obj.scale = (scale_factor, scale_factor, scale_factor)
            obj.location = (0, 0, 0)
            
            # Apply transformations
            bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)

# Function to render the object from different angles and create a GIF
def render_gif(obj_path, output_path):
    import_object(obj_path)
    setup_camera()
    setup_lighting()
    
    camera = bpy.context.scene.camera

    # Set up render settings
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    
    # Create a folder for the frames
    frame_folder = os.path.join(output_path, 'frames')
    os.makedirs(frame_folder, exist_ok=True)
    
    # Render frames
    for frame in range(36):
        angle = frame * 10  # 10 degrees per frame
        camera.location.x = 5 * math.cos(math.radians(angle))
        camera.location.y = 5 * math.sin(math.radians(angle))
        camera.location.z = 3
        camera.rotation_euler = (math.radians(60), 0, math.radians(angle + 90))
        bpy.context.scene.render.filepath = os.path.join(frame_folder, f'frame_{frame:02d}.png')
        bpy.ops.render.render(write_still=True)
    
    # Create a GIF from the rendered frames
    os.system(f"convert -delay 10 -loop 0 {frame_folder}/frame_*.png {output_path}.gif")
    
    # Clean up the frames
    for file in os.listdir(frame_folder):
        os.remove(os.path.join(frame_folder, file))
    os.rmdir(frame_folder)

# Function to traverse the directory structure and process OBJ files
def traverse_directories(base_dir: str = "/pub0/daniel/Complete_version_3/Blender_files", output_dir: str = "/pub0/daniel/Complete_version_3/rendered_gifs"):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".obj"):
                obj_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, base_dir)
                
                # Use regex to replace brackets with underscores and spaces with underscores in the output path
                safe_rel_path = re.sub(r'[\(\)\s]', '_', rel_path)
                
                # Also handle spaces in the filename
                safe_filename = re.sub(r'[\(\)\s]', '_', os.path.splitext(file)[0])
                
                output_path = os.path.join(output_dir, safe_rel_path, safe_filename)
                
                # Check if the GIF already exists
                gif_path = f"{output_path}.gif"
                if os.path.exists(gif_path):
                    print(f"Skipping {obj_path} as {gif_path} already exists")
                    continue
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                print(f"Rendering {obj_path} to {output_path}")
                render_gif(obj_path, output_path)

# Run the script
if __name__ == "__main__":
    tyro.cli(traverse_directories)