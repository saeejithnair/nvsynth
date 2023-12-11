#!/bin/bash

# Check if the image directory is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <image_directory>"
    exit 1
fi

IMG_DIR=$1
DIR_NAME=$(basename "$IMG_DIR")
VIEWPORTS=12  # Assuming 12 viewpoints as per your description
FRAMERATE=15  # Set this to the desired frames per second

# Loop through each viewport
for (( v=1; v<=VIEWPORTS; v++ ))
do
    # Generate a pattern for the current viewport
    PATTERN="${IMG_DIR}/scene_*/????_viewport_${v}.png"

    # Check if we have images to process
    if compgen -G "$PATTERN" > /dev/null; then
        echo "Found images for viewport $v."
    else
        echo "No images found for viewport $v."
        continue
    fi

    # Create a GIF for this viewport using FFmpeg
    GIF_NAME="${DIR_NAME}_viewport_${v}.gif"
    PALETTE="${DIR_NAME}_palette_${v}.png"

    # Use FFmpeg to create the palette
    ffmpeg -framerate $FRAMERATE -pattern_type glob -i "$PATTERN" -vf "scale=640:-1:flags=lanczos,palettegen" -y "$PALETTE"

    # Use FFmpeg to create the GIF with the label
    ffmpeg -framerate $FRAMERATE -pattern_type glob -i "$PATTERN" -i "$PALETTE" -filter_complex "[0:v]scale=640:-1:flags=lanczos,drawtext=fontfile=/path/to/font.ttf:text='Camera ${v}':x=W-tw-10:y=10:fontsize=24:fontcolor=white@0.8:shadowx=2:shadowy=2,paletteuse" -y "$GIF_NAME"

    # Remove the temporary palette file
    if [ -f "$PALETTE" ]; then
        rm "$PALETTE"
    fi

    echo "Created GIF for viewport $v: $GIF_NAME"
done

echo "All GIFs created successfully."
