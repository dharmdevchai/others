import os
import cv2
from pyzbar.pyzbar import decode

# Replace this with the actual path to your dataset folder on your computer
# Example: dataset_path = r"C:\Users\YourName\Datasets\barcode-and-qr"
dataset_path = "path/to/your/local/dataset"

def extract_codes_from_dataset(directory):
    print("Starting barcode and QR code extraction...\n")
    success_count = 0
    total_images = 0

    if not os.path.exists(directory):
        print(f"Error: The directory '{directory}' does not exist. Please check your path.")
        return

    for root, dirs, files in os.walk(directory):
        for file in files:
            # Process common image formats
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                total_images += 1
                image_path = os.path.join(root, file)
                
                # Read the image using OpenCV
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # Decode barcodes and QR codes
                decoded_objects = decode(image)
                
                if decoded_objects:
                    for obj in decoded_objects:
                        success_count += 1
                        code_type = obj.type
                        code_data = obj.data.decode('utf-8')
                        print(f"[{code_type}] Found in {file}:")
                        print(f" -> Extracted Text: {code_data}\n")

    print(f"--- Extraction Complete ---")
    print(f"Scanned {total_images} images. Successfully decoded {success_count} codes.")

# Run the function
if __name__ == "__main__":
    extract_codes_from_dataset(dataset_path)
