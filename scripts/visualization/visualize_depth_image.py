#%%
import numpy as np
import matplotlib.pyplot as plt

# %%

if __name__ == "__main__":
    image_path = "/home/smnair/work/nutrition/nvsynth/generated/_fv_output_1717445954/depth_images/scene_0000/0000_viewport_12.npy"

    depth_image = np.load(image_path)
    image_path = "/home/smnair/work/nutrition/nvsynth/generated/_fv_output_1717445954/depth_images/scene_0000/0000_viewport_12.npy"

    depth_image = np.load(image_path)

    # Visualizing the depth image
    plt.imshow(depth_image, cmap='viridis')
    plt.axis('off')  # Turn off the axis
    # plt.colorbar()  # Adds a colorbar to check the depth scale
    plt.savefig('depth_image.png', pad_inches=0.0, bbox_inches='tight')
    plt.close()