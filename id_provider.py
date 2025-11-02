# folder_path = "C:\Users\Sumit\OneDrive\Pictures\Screenshots"
# 
#  
import os
import uuid
from PIL import Image
from pathlib import Path


def generate_unique_id():
    return str(uuid.uuid4())

def save_image_with_id(image_path, save_directory):
    # Ensure the save directory exists
    Path(save_directory).mkdir(parents=True, exist_ok=True)

    # Open the image
    print(f"Processing image: {image_path}")
    with Image.open(image_path) as img:
        # Generate a unique ID
        unique_id = generate_unique_id()
        
        # Create a new filename with the unique ID
        # file_extension = os.path.splitext(image_path)[1]
        new_filename = f"{unique_id}.png"
        save_path = os.path.join(save_directory, new_filename)
        
        # Save the image with the new filename
        img.save(save_path)
        
    return unique_id, save_path

# Example usage
if __name__ == "__main__":


    folder_path = "E:\HXEFH" # your image folder
    save_directory = "C:\\Users\\Sumit\\OneDrive\\Desktop\\image_vision\\uuid_images" # your save folder

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            unique_id, saved_path = save_image_with_id(image_path, save_directory)
            print(f"Image saved with ID: {unique_id} at {saved_path}")