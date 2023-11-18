# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 21:26:55 2023

@author: kahju
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = ['001.png','002.png', '003.png', '004.png', '005.png', '006.png', '007.png', '008.png']
for i in range(len(image_path)):
    image = cv2.imread(image_path[i], cv2.IMREAD_GRAYSCALE)
# Check if the image was loaded correctly
if image is None:
    print('Error: Image did not load.')
else:
    # Convert the image to binary using Otsu's thresholding
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Calculate the horizontal projection histogram
    horizontal_projection = np.sum(binary_image, axis=1)

    # Calculate the vertical projection histogram
    vertical_projection = np.sum(binary_image, axis=0)

    # Plot the histograms and the binary image
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # Plot binary image
    ax1.imshow(binary_image, cmap='gray')
    ax1.set_title('Binary Image')
    ax1.axis('off')

    # Plot horizontal histogram
    ax2.barh(range(len(horizontal_projection)), horizontal_projection, height=1)
    ax2.set_title('Horizontal Projection Histogram')
    ax2.invert_yaxis()  # Invert y-axis to match the image

    # Plot vertical histogram
    ax3.bar(range(len(vertical_projection)), vertical_projection)
    ax3.set_title('Vertical Projection Histogram')

    # Show the plots
    plt.tight_layout()
    plt.show()
    
    def extract_actual_paragraphs(binary_img, horizontal_proj, min_separation=20):
        
        """ Extract actual paragraphs from the binary image by considering a minimum separation
        between paragraphs.
        """
        # Threshold to find lines: any non-zero value in the horizontal projection
        line_indices = np.nonzero(horizontal_proj)[0]
    
        # Initialize lists to hold the start and end of paragraphs
        start_indices = [line_indices[0]]
        end_indices = []
    
        # Loop through line indices to find paragraph breaks
        for i in range(1, len(line_indices)):
            # Check for a gap between lines greater than min_separation
            if line_indices[i] - line_indices[i-1] > min_separation:
                end_indices.append(line_indices[i-1])
                start_indices.append(line_indices[i])
    
        # Add the last line index as the end of the last paragraph
        end_indices.append(line_indices[-1])
    
        # Extract paragraphs using the start and end indices
        paragraphs = [binary_img[start:end+1] for start, end in zip(start_indices, end_indices)]
    
        return paragraphs
    
    # Apply the function to extract actual paragraphs
    actual_paragraphs = extract_actual_paragraphs(binary_image, horizontal_projection)
    
    # Save the actual paragraphs as image files
    actual_paragraph_image_paths = []
    for i, paragraph_img in enumerate(actual_paragraphs):
        actual_paragraph_path = f'001output/paragraph{i+1}.png'
        cv2.imwrite(actual_paragraph_path, 255 - paragraph_img)  # Invert to save as original
        actual_paragraph_image_paths.append(actual_paragraph_path)
    
    # Output the paths of the saved actual paragraph images
    actual_paragraph_image_paths

