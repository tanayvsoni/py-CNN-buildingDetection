import os
import shutil
from PIL import Image

def resize_image(image_path, output_path):
    # Open the image file
    image = Image.open(image_path)
    
    # Resize the image to 1000x1000 without maintaining aspect ratio
    resized_image = image.resize((1000, 1000))
    os.remove(image_path)
    
    # Save the resized image
    resized_image.save(output_path)


def org_files(data_directory):
    """Prompts user to input image data and moves said image into zip file and stores
    relevent data into image_data.csv file.

    Args:
        data_directory (string): location of data
    """
    
    # Get last image ID that was used from csv file
    with open(f'{data_directory}/output/image_data.csv','r') as csvfile:
        file = csvfile.read().splitlines()
        last_imgID = int(file[-1].split(',')[0][:-4])
        
    # Get all input images (should be only 1 image ideally)      
    img_names = os.listdir(f'{data_directory}/input')
    
    for img in img_names:
        # Create new image name by adding 1 to previous ID number
        last_imgID += 1
        new_name = f'{last_imgID}.png'
        
        resize_image(f'{data_directory}/input/{img}',f'{data_directory}/input/{img}')
        
        # Rename Image
        os.rename(f'{data_directory}/input/{img}',f'{data_directory}/input/{new_name}')
        
        # Get relevent data
        print(f'\nFor image {img} please enter relevent info:\n')
                
        country_name = input('Enter name of country present in image: ').lower()
        num_building = input('Enter number of buildings present in image: ')
        comments = input('Any extra comments? Leave blank if none: ')
        
        # Write data input csv file    
        with open(f'{data_directory}/output/image_data.csv','a') as csvfile:
            csvfile.write(f'{new_name},{num_building},{country_name},{comments}\n')
        
        shutil.move(f'{data_directory}/input/{new_name}', f'{data_directory}/output/{new_name}')
 
def main():
    while True:
        org_files('./data')
        if input('\nAll files have been stored, would you like to input again (y/n): ').lower() == 'n': break
                       
if __name__ == '__main__':
    main()
    
    