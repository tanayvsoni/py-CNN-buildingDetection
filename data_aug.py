import os
from PIL import Image
import csv
import threading


def generate_new_image_names(image_name):
    """
    Generate new image names based on the original image name and the number of transformations.
    
    Args:
        image_name (str): Original image name.
        
    Returns:
        list: List of new image names with transformations.
    """
    
    base_name, extension = os.path.splitext(image_name)
    new_names = []
    
    for i in range(num_transforms):
        new_name = f'{base_name}_{i + 1}.png'
        new_names.append(new_name)
    
    return new_names

def apply_rotation(image, angle):
    """
    Apply rotation transformation to an image.
    
    Args:
        image (PIL.Image): Original image.
        angle (int): Rotation angle in degrees.
        
    Returns:
        PIL.Image: Transformed image.
    """
    return image.rotate(angle)

def apply_horizontal_flip(image):
    """
    Apply horizontal flip transformation to an image.
    
    Args:
        image (PIL.Image): Original image.
        
    Returns:
        PIL.Image: Transformed image.
    """
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def apply_vertical_flip(image):
    """
    Apply vertical flip transformation to an image.
    
    Args:
        image (PIL.Image): Original image.
        
    Returns:
        PIL.Image: Transformed image.
    """
    return image.transpose(Image.FLIP_TOP_BOTTOM)

def save_transformed_image(new_name, transformed_image, label):
    """
    Save a transformed image and update the CSV file.
    
    Args:
        new_name (str): New image name.
        transformed_image (PIL.Image): Transformed image.
        label (str): Image label.
    """
    
    new_image_path = os.path.join(output_folder, new_name)
    transformed_image.save(new_image_path)
    print(f"{new_name} Saved")
    
    with open(new_csv, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([new_name, label])

def process_image_row(row):
    """
    Process a row of image data, perform transformations, and save the results.
    
    Args:
        row (list): List containing image name and label.
    """
    
    image_name = row[0]
    label = row[1]
    image_path = os.path.join(input_folder, image_name)
    

    new_image_names = generate_new_image_names(image_name)
    original_image = Image.open(image_path)

    for i in range(num_rotations):
        new_image = apply_rotation(original_image.copy(), 90 * i)
        save_transformed_image(new_image_names[0], new_image, label)
        new_image_names = new_image_names[1:]

    new_image = apply_horizontal_flip(original_image.copy())

    for i in range(num_rotations):
        new_image = apply_rotation(new_image, 90 * i)
        save_transformed_image(new_image_names[0], new_image, label)
        new_image_names = new_image_names[1:]

def main():
    # Global variables
    global input_folder, output_folder, num_transforms, num_rotations, new_csv

    # Define paths and parameters
    input_folder = './data/output'
    output_folder = './data/data_aug_out'
    new_csv = './data/new_image_data.csv'
    old_csv = './data/image_data.csv'
    num_transforms = 8
    num_rotations = 4
    
    # Process image data and apply transformations using threads
    with open(old_csv, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        threads = []

        for row in csv_reader:
            thread = threading.Thread(target=process_image_row, args=(row,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

    print("Image transformations and CSV update complete.")

if __name__ == "__main__":
    main()