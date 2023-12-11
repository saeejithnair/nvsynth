#!/usr/bin/env python3

import csv
import json
import argparse
import os

import pandas as pd
import time


# ------------ STEP 0: START TIMER --------------
# Start the timer and print out log that script started.
start_time = time.time()
print('Starting script to generate dish metadata file...')


# ------------ STEP 0: READ INPUTS --------------
# Create the parser
parser = argparse.ArgumentParser() 

# Read in arguments for input.
parser.add_argument('--metadata_dir', 
                    help='Location to the directory containing the metadata json files', 
                    type=str, 
                    required=True) 
parser.add_argument('--ingredients_metadata_file', 
                    help='Location to the ingredients metadata csv file', 
                    type=str, 
                    required=True) 


# Parse the arguments
args = parser.parse_args()
metadata_dir = args.metadata_dir
ingredients_metadata_file = args.ingredients_metadata_file

df_dish_metadata = pd.DataFrame(
    columns=['dish_id', 'total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein', 'num_ingrs', 
    'ingr_1_id', 'ingr_1_name', 'ingr_1_grams', 'ingr_1_calories', 'ingr_1_fat', 'ingr_1_carb', 'ingr_1_protein',
    'ingr_2_id', 'ingr_2_name', 'ingr_2_grams', 'ingr_2_calories', 'ingr_2_fat', 'ingr_2_carb', 'ingr_2_protein',
    'ingr_3_id', 'ingr_3_name', 'ingr_3_grams', 'ingr_3_calories', 'ingr_3_fat', 'ingr_3_carb', 'ingr_3_protein',
    'ingr_4_id', 'ingr_4_name', 'ingr_4_grams', 'ingr_4_calories', 'ingr_4_fat', 'ingr_4_carb', 'ingr_4_protein',
    'ingr_5_id', 'ingr_5_name', 'ingr_5_grams', 'ingr_5_calories', 'ingr_5_fat', 'ingr_5_carb', 'ingr_5_protein',
    ])

print("metadata_dir = ", metadata_dir)
print("ingredients_metadata_file = ", ingredients_metadata_file)

# ------------ STEP 1: READ IN METADATA FILES --------------
scene_list = [ f.name for f in os.scandir("test_metadata") if f.is_dir() ]
df_ingredients = pd.read_csv(ingredients_metadata_file)

# Loop through all the scenes in the metadata folder
for scene in scene_list:
    scene_id = scene.split('_')[1]

    if os.path.isfile(f'{metadata_dir}/scene_{scene_id}/{scene_id}_viewport_1.json'):
      print("Processing for scene = ", scene)

      with open(f'{metadata_dir}/scene_{scene_id}/{scene_id}_viewport_1.json') as json_file:
          data = json.load(json_file)

      # Create a dictionary to store the nutrition information for this scene.
      nutrition_dict = {}
      nutrition_dict["dish_id"] = f"dish_{scene_id}"

      # Placeholders for scene metrics (total calories, num_ingrs, ...).
      nutrition_dict["total_calories"] = 0
      nutrition_dict["total_mass"] = 0
      nutrition_dict["total_fat"] = 0
      nutrition_dict["total_carb"] = 0
      nutrition_dict["total_protein"] = 0
      nutrition_dict["num_ingrs"] = 0

      ingr_counter = 0

      # Loop through the food items in the scene.
      for key, value in data.items():
          ingr = value["class"]

          if "food_id" in ingr:
              # Update the number of ingredients.
              ingr_counter = ingr_counter + 1
              nutrition_dict["num_ingrs"] = ingr_counter

              # Get the ingredient name.
              ingr_name = ingr[5:].replace("_", "-")

              # Find the information for this ingredient in the ingredients metadata.
              df_ingr_info = df_ingredients[df_ingredients["ingr"] == ingr_name]

              # Save the ingredient information for this ingredient.
              nutrition_dict[f"ingr_{ingr_counter}_id"] = df_ingr_info["id"].values[0]
              nutrition_dict[f"ingr_{ingr_counter}_name"] = df_ingr_info["ingr"].values[0]
              nutrition_dict[f"ingr_{ingr_counter}_grams"] = df_ingr_info["weight(g)"].values[0]
              nutrition_dict[f"ingr_{ingr_counter}_calories"] = df_ingr_info["cal(kCal)"].values[0]
              nutrition_dict[f"ingr_{ingr_counter}_fat"] = df_ingr_info["fat(g)"].values[0]
              nutrition_dict[f"ingr_{ingr_counter}_carb"] = df_ingr_info["carb(g)"].values[0]
              nutrition_dict[f"ingr_{ingr_counter}_protein"] = df_ingr_info["protein(g)"].values[0]

              # Update the scene nutrititional information.
              nutrition_dict["total_calories"] = nutrition_dict["total_calories"] + df_ingr_info["cal(kCal)"].values[0] 
              nutrition_dict["total_mass"] = nutrition_dict["total_mass"] + df_ingr_info["weight(g)"].values[0]
              nutrition_dict["total_fat"] = nutrition_dict["total_fat"] + df_ingr_info["fat(g)"].values[0]
              nutrition_dict["total_carb"] = nutrition_dict["total_carb"] + df_ingr_info["carb(g)"].values[0]
              nutrition_dict["total_protein"] = nutrition_dict["total_protein"] + df_ingr_info["protein(g)"].values[0]
      
      # Fill in the rest of the columns with NA if there are less than 5 ingredients.
      while ingr_counter < 5:
          ingr_counter = ingr_counter + 1
          nutrition_dict[f"ingr_{ingr_counter}_id"] = "NA"
          nutrition_dict[f"ingr_{ingr_counter}_name"] = "NA"
          nutrition_dict[f"ingr_{ingr_counter}_grams"] = "NA"
          nutrition_dict[f"ingr_{ingr_counter}_calories"] = "NA"
          nutrition_dict[f"ingr_{ingr_counter}_fat"] = "NA"
          nutrition_dict[f"ingr_{ingr_counter}_carb"] = "NA"
          nutrition_dict[f"ingr_{ingr_counter}_protein"] = "NA"

      df_dish_metadata = df_dish_metadata.append(nutrition_dict, ignore_index=True)

df_dish_metadata.to_csv("dish_metadata.csv", index=False)


# ------------ STEP 0: END TIMER --------------
# Stop the timer and print out time to run.
print('Script to generate dish metadata file complete!')
print("--- %s seconds ---" % (time.time() - start_time))