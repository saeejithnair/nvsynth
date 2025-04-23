#!/usr/bin/env python3
import os
import shutil
import csv
import re
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_conversion.log"),
        logging.StreamHandler()
    ]
)

# Paths
SOURCE_DIR = '/pub0/smnair/nutrition/Blender_files'
DEST_DIR = '/pub0/smnair/nutrition/dataset_0425'
CSV_PATH = 'configs/MetaFood3D_nutrition_v2.csv'
SCALE_FACTORS_PATH = '/pub0/smnair/nutrition/dataset_0425/scale_factors.csv'

# Create destination directory if it doesn't exist
os.makedirs(DEST_DIR, exist_ok=True)

# Helper function to normalize names for comparison
def normalize_name(name):
    # Convert to lowercase, replace special chars with underscores
    norm = name.lower()
    # Remove "new_" prefix if it exists
    if norm.startswith('new_'):
        norm = norm[4:]
    # Convert remaining characters
    norm = re.sub(r'[^a-z0-9]', '_', norm)
    # Replace multiple underscores with a single one
    norm = re.sub(r'_+', '_', norm)
    # Remove leading/trailing underscores
    norm = norm.strip('_')
    return norm

# Load the nutrition data from CSV
food_data = {}
normalized_mapping = {}  # Maps normalized names to original keys
csv_entries_count = 0
scale_factors = []  # To store the scale factors for each processed item

# Debug lists
csv_keys = []
directory_keys = []

with open(CSV_PATH, 'r') as csvfile:
    reader = csv.reader(csvfile)
    # Skip the header row explicitly
    next(reader)
    for i, row in enumerate(reader, 1):
        if len(row) >= 5:  # Ensure row has enough columns
            category, object_name, _, weight, *_ = row
            
            # Skip any other potential header rows
            if category == 'Food_Category' or weight == 'Weight' or category == 'Object_name':
                continue
                
            try:
                weight_g = float(weight)
                csv_entries_count += 1
                
                # Clean category name for the new folder name - replace spaces with hyphens
                clean_category = category.lower().replace('(', '-').replace(')', '-').replace('_', '-').replace(' ', '-')
                clean_category = re.sub(r'-+', '-', clean_category)  # Replace multiple hyphens with single hyphen
                clean_category = clean_category.rstrip('-')  # Remove trailing hyphens
                
                # Create the new folder name in the target format
                new_folder_name = f"id-{csv_entries_count}-{clean_category}-{int(weight_g)}g"
                
                # Create the model_label for scale_factors.csv
                model_label = new_folder_name.replace('-', '_')
                
                # Store both original case and lowercase versions of keys
                key_original = (category, object_name)
                key_lower = (category.lower(), object_name.lower())
                
                # Track the keys for debugging
                csv_keys.append(key_original)
                csv_keys.append(key_lower)
                
                # Store the mapping for both versions to improve exact match rate
                food_data[key_original] = {
                    'weight': weight_g,
                    'new_name': new_folder_name,
                    'id': csv_entries_count,
                    'model_label': model_label
                }
                
                # Also add lowercase version to increase exact match chances
                if key_lower != key_original:
                    food_data[key_lower] = {
                        'weight': weight_g,
                        'new_name': new_folder_name,
                        'id': csv_entries_count,
                        'model_label': model_label
                    }
                
                # Create normalized key for fuzzy matching
                norm_category = normalize_name(category)
                norm_object = normalize_name(object_name)
                normalized_key = (norm_category, norm_object)
                normalized_mapping[normalized_key] = key_original
                
            except ValueError:
                logging.warning(f"Could not parse weight for {category}/{object_name}")

logging.info(f"Loaded {csv_entries_count} entries from CSV file")

# Special case handling for problematic entries
# Add carrot_8 mapping if it's missing
carrot_entries = [(cat, obj) for (cat, obj) in food_data.keys() if cat.lower() == 'carrot' and obj.lower() == 'carrot_8']
if not carrot_entries:
    logging.info("Adding special case entry for Carrot/carrot_8")
    # Find similar entry to base the new one on
    carrot_template = None
    for key in food_data.keys():
        if key[0].lower() == 'carrot' and key[1].lower().startswith('carrot_'):
            carrot_template = key
            break
    
    if carrot_template:
        template_data = food_data[carrot_template].copy()
        csv_entries_count += 1
        # Create new entry with a default weight of the template
        new_key = ('Carrot', 'carrot_8')
        # Create the model_label for scale_factors.csv
        model_label = f"id_{csv_entries_count}_carrot_{int(template_data['weight'])}g"
        food_data[new_key] = {
            'weight': template_data['weight'],
            'new_name': f"id-{csv_entries_count}-carrot-{int(template_data['weight'])}g",
            'id': csv_entries_count,
            'model_label': model_label
        }
        # Add lowercase version too
        food_data[('carrot', 'carrot_8')] = food_data[new_key]
        # Add to normalized mapping
        normalized_mapping[('carrot', 'carrot_8')] = new_key

# Create stats counters
exact_matches = 0
normalized_matches = 0
fuzzy_matches = 0
unmatched = 0
processed_folders = set()
skipped_existing = 0

# Debug: print some examples of the loaded data
logging.debug(f"Example CSV entries:")
for i, key in enumerate(list(food_data.keys())[:5]):
    logging.debug(f"  {key}: {food_data[key]['new_name']}")

# Process the source directory
for root, dirs, files in os.walk(SOURCE_DIR):
    rel_path = os.path.relpath(root, SOURCE_DIR)
    
    # Skip the top-level directory
    if rel_path == '.':
        continue
    
    # Extract category and object name from the path
    parts = rel_path.split(os.sep)
    if len(parts) == 2:  # We're in a subfolder like Whole_Chicken/Chicken_5
        category, object_name = parts
        
        # Track for debugging
        directory_keys.append((category, object_name))
        directory_keys.append((category.lower(), object_name.lower()))
        
        # Track if we've processed this folder
        folder_key = f"{category}/{object_name}"
        
        # Try several variations for exact match
        exact_match_keys = [
            (category, object_name),               # Exact original case
            (category.lower(), object_name.lower()) # Lowercase both
        ]
        
        found_exact = False
        for key in exact_match_keys:
            if key in food_data:
                item_data = food_data[key]
                new_folder_name = item_data['new_name']
                model_label = item_data['model_label']
                
                # Check if the destination directory already exists
                dest_path = os.path.join(DEST_DIR, new_folder_name)
                if os.path.exists(dest_path):
                    logging.info(f"Skipping {rel_path} → {new_folder_name} (already exists)")
                    processed_folders.add(folder_key)
                    skipped_existing += 1
                    # Add to scale factors even if skipped (for completeness)
                    scale_factors.append((model_label, 1.0))
                    found_exact = True
                    break
                
                # Create destination directory
                os.makedirs(dest_path, exist_ok=True)
                
                # Copy all files from source to destination
                for file in files:
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(dest_path, file)
                    shutil.copy2(src_file, dst_file)
                
                # Also copy the textures folder if it exists
                textures_dir = os.path.join(root, 'textures')
                if os.path.exists(textures_dir):
                    textures_dest = os.path.join(dest_path, 'textures')
                    if not os.path.exists(textures_dest):
                        shutil.copytree(textures_dir, textures_dest)
                
                logging.info(f"Copied {rel_path} → {new_folder_name}")
                processed_folders.add(folder_key)
                exact_matches += 1
                # Add to scale factors
                scale_factors.append((model_label, 1.0))
                found_exact = True
                break
        
        if not found_exact:
            # Try normalized matching
            norm_category = normalize_name(category)
            norm_object = normalize_name(object_name)
            normalized_key = (norm_category, norm_object)
            
            # Check if we have a normalized match
            if normalized_key in normalized_mapping:
                original_key = normalized_mapping[normalized_key]
                item_data = food_data[original_key]
                new_folder_name = item_data['new_name']
                model_label = item_data['model_label']
                
                # Check if the destination directory already exists
                dest_path = os.path.join(DEST_DIR, new_folder_name)
                if os.path.exists(dest_path):
                    logging.info(f"Skipping {rel_path} → {new_folder_name} (already exists)")
                    processed_folders.add(folder_key)
                    skipped_existing += 1
                    # Add to scale factors even if skipped (for completeness)
                    scale_factors.append((model_label, 1.0))
                    continue
                
                # Create destination directory
                os.makedirs(dest_path, exist_ok=True)
                
                # Copy all files from source to destination
                for file in files:
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(dest_path, file)
                    shutil.copy2(src_file, dst_file)
                
                # Also copy the textures folder if it exists
                textures_dir = os.path.join(root, 'textures')
                if os.path.exists(textures_dir):
                    textures_dest = os.path.join(dest_path, 'textures')
                    if not os.path.exists(textures_dest):
                        shutil.copytree(textures_dir, textures_dest)
                
                logging.info(f"Copied {rel_path} → {new_folder_name}")
                processed_folders.add(folder_key)
                normalized_matches += 1
                # Add to scale factors
                scale_factors.append((model_label, 1.0))
            else:
                # Try finding a match by comparing variations of the name
                found = False
                for (csv_category, csv_object), data in food_data.items():
                    if normalize_name(csv_category) == norm_category:
                        # Try variations like with/without prefix, different casing, etc.
                        csv_obj_base = normalize_name(csv_object)
                        obj_name_base = norm_object
                        
                        # Try different naming patterns
                        if (csv_obj_base == obj_name_base or
                            csv_obj_base.replace('_', '') == obj_name_base.replace('_', '') or
                            csv_obj_base + '_1' == obj_name_base or
                            obj_name_base + '_1' == csv_obj_base):
                            
                            new_folder_name = data['new_name']
                            model_label = data['model_label']
                            
                            # Check if the destination directory already exists
                            dest_path = os.path.join(DEST_DIR, new_folder_name)
                            if os.path.exists(dest_path):
                                logging.info(f"Skipping {rel_path} → {new_folder_name} (already exists)")
                                processed_folders.add(folder_key)
                                skipped_existing += 1
                                # Add to scale factors even if skipped (for completeness)
                                scale_factors.append((model_label, 1.0))
                                found = True
                                break
                            
                            # Create destination directory
                            os.makedirs(dest_path, exist_ok=True)
                            
                            # Copy all files
                            for file in files:
                                src_file = os.path.join(root, file)
                                dst_file = os.path.join(dest_path, file)
                                shutil.copy2(src_file, dst_file)
                            
                            # Copy textures folder
                            textures_dir = os.path.join(root, 'textures')
                            if os.path.exists(textures_dir):
                                textures_dest = os.path.join(dest_path, 'textures')
                                if not os.path.exists(textures_dest):
                                    shutil.copytree(textures_dir, textures_dest)
                            
                            logging.info(f"Copied {rel_path} → {new_folder_name} (fuzzy match)")
                            processed_folders.add(folder_key)
                            fuzzy_matches += 1
                            # Add to scale factors
                            scale_factors.append((model_label, 1.0))
                            found = True
                            break
                
                if not found:
                    logging.warning(f"No data found for {category}/{object_name}")
                    unmatched += 1

# Check for CSV entries that weren't used
unused_csv_entries = 0
unique_keys = set()
for (cat, obj) in food_data.keys():
    # Only count each unique item once (not both case variations)
    key = (cat.lower(), obj.lower())
    if key in unique_keys:
        continue
    unique_keys.add(key)
    
    folder_key = f"{cat}/{obj}"
    if folder_key not in processed_folders and f"{cat.lower()}/{obj.lower()}" not in [f.lower() for f in processed_folders]:
        logging.warning(f"CSV entry not used: {cat}/{obj}")
        unused_csv_entries += 1

# Write the scale factors CSV
with open(SCALE_FACTORS_PATH, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['model_label', 'scale'])
    for model_label, scale in scale_factors:
        writer.writerow([model_label, scale])

# Summary
logging.info(f"Dataset conversion completed.")
logging.info(f"Statistics:")
logging.info(f"  - Total CSV entries: {csv_entries_count}")
logging.info(f"  - Exact matches: {exact_matches}")
logging.info(f"  - Normalized matches: {normalized_matches}")
logging.info(f"  - Fuzzy matches: {fuzzy_matches}")
logging.info(f"  - Skipped (already exist): {skipped_existing}")
logging.info(f"  - Unmatched folders: {unmatched}")
logging.info(f"  - Unused CSV entries: {unused_csv_entries}")
logging.info(f"  - Total folders processed: {exact_matches + normalized_matches + fuzzy_matches}")
logging.info(f"  - Scale factors CSV written to: {SCALE_FACTORS_PATH}")

print("Dataset conversion completed. See dataset_conversion.log for details.")
