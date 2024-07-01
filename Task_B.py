import cv2
import numpy as np
import os
import shutil

def reset_dir():
    for i in range(1,9):
        try:
            shutil.rmtree('00' + str(i) + "_output")
        except FileNotFoundError:
            pass
        except Exception:
            pass

def vertical_projection(binary_image):
    # Sum up the pixels column-wise
    return np.sum(binary_image, axis=0)

def horizontal_projection(binary_image):
    # Sum up the pixels row-wise
    return np.sum(binary_image, axis=1)

def detect_columns(vertical_projection, threshold, min_column_width):
    columns = []
    in_column = False
    start_index = 0

    for index, value in enumerate(vertical_projection):
        if value > threshold and not in_column:
            in_column = True
            start_index = index
        elif value <= threshold and in_column:
            in_column = False
            if index - start_index > min_column_width:  # Filter out narrow columns
                columns.append((start_index, index))
    
    if in_column and len(vertical_projection) - start_index > min_column_width:
        columns.append((start_index, len(vertical_projection)))

    return columns

def split_binary_image_with_skip(binary_image, skip_rows):
    # Find the height of the binary image
    height = binary_image.shape[0]

    # Calculate the number of rows to skip
    skip_rows_end = min(height, skip_rows)

    # Split the image into two parts: top part (skipped rows) and bottom part (without skipped rows)
    top_part = binary_image[:skip_rows_end, :]
    bottom_part = binary_image[skip_rows_end:, :]

    return top_part, bottom_part

# Function to extract actual paragraphs from the binary image by considering a minimum separation
# between paragraphs.
def extract_actual_paragraphs(binary_img, horizontal_proj, min_separation=25):
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

# Function to process each image and extract paragraphs
def process_image(i, image_num, column_threshold, min_column_width):
    image_path = f'{image_num}.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded correctly
    if image is None:
        return f'Error: Image {image_num}.png did not load.'

    # Convert the image to binary using Otsu's thresholding
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if i == 4:
        _, binary_image = split_binary_image_with_skip(binary_image, 400)

    # Vertical projection for column detection
    v_proj = vertical_projection(binary_image)
    column_boundaries = detect_columns(v_proj, column_threshold, min_column_width)

    actual_paragraph_image_paths = []
    # Process each column to extract paragraphs
    for i, (start_col, end_col) in enumerate(column_boundaries):
        column_image = binary_image[:, start_col:end_col]
        # Calculate the horizontal projection histogram
        h_proj = horizontal_projection(column_image)
        # Apply the function to extract actual paragraphs
        actual_paragraphs = extract_actual_paragraphs(column_image, h_proj)

        # Create output directory
        output_dir = f'{image_num}_output/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save paragraphs as images
        column_dir = os.path.join(output_dir, f'column_{i+1}')
        os.makedirs(column_dir, exist_ok=True)
        
        for j, paragraph_img in enumerate(actual_paragraphs):
            paragraph_path = os.path.join(column_dir, f'paragraph_{j+1}.png')
            cv2.imwrite(paragraph_path, 255 - paragraph_img)  # Invert image to original before saving
            actual_paragraph_image_paths.append(paragraph_path)

    return f'Processed and saved paragraphs for {actual_paragraph_image_paths}'

def contains_text(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, binary_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Invert the binary image
    inverted_binary_image = cv2.bitwise_not(binary_image)
    
    # Find contours in the inverted binary image
    contours, _ = cv2.findContours(inverted_binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If there are contours, there is likely text
    return len(contours) > 0

def contains_table(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, binary_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Invert the binary image
    inverted_binary_image = cv2.bitwise_not(binary_image)
    
    # Find contours in the inverted binary image
    contours, _ = cv2.findContours(inverted_binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through contours and check if they have a large area (potential table)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # You may need to adjust this threshold based on your images
            return True
    
    return False

def contains_picture(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute the mean color in the image
    mean_color = cv2.mean(image)[:3]
    
    # Set a threshold for color intensity (you may need to adjust this threshold)
    threshold = 100
    
    # Check if the mean color intensity is below the threshold
    return all(val < threshold for val in mean_color)

reset_dir()
# Process all images and extract paragraphs
all_paragraphs_paths = {}
for i in range(1, 9):
    image_num = f'00{i}'
    paragraph_paths = process_image(i, image_num, column_threshold=10, min_column_width=10)
    all_paragraphs_paths[image_num] = paragraph_paths
    print(f"Saving results to {image_num}_output/")

# Delete images with only tables or pictures
for i in range(1, 9):
    output_path = f"00{i}_output"
    for root, dirs, files in os.walk(output_path):
        for filename in files:
            imagepath = os.path.join(root, filename)

            has_text = contains_text(imagepath)
            has_table = contains_table(imagepath)
            has_picture = contains_picture(imagepath)

            if has_table or has_picture:
                os.remove(imagepath)
                print(f"Deleted: {imagepath}")