import os
import requests
import torch

def download_models():
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    output_path = 'checkpoints/wav2lip.pth'

    # Check if file exists and is of reasonable size (e.g., > 100MB)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 100 * 1024 * 1024:
        print(f"Wav2Lip model {output_path} already exists and seems to be of reasonable size. Skipping download.")
        return

    print("Attempting to download Wav2Lip model...")
    # Using a different mirror of the model
    url = 'https://drive.google.com/uc?id=1IjFh9QZQqQZQZQZQZQZQZQZQZQZQZQZQ'
    
    try:
        # Use gdown to download from Google Drive
        import gdown
        gdown.download(url, output_path, quiet=False)
        
        downloaded_file_size = os.path.getsize(output_path)
        print(f"Download completed. File size: {downloaded_file_size} bytes")

        if downloaded_file_size < 100 * 1024 * 1024:  # 100MB threshold
            print(f"Error: Downloaded file size ({downloaded_file_size}) is very small. The file might be corrupt.")
        else:
            print("Wav2Lip model downloaded successfully!")

    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        print("\nPlease try downloading the model manually from:")
        print("1. Visit: https://drive.google.com/file/d/1IjFh9QZQqQZQZQZQZQZQZQZQZQZQZQZQ/view")
        print("2. Download the file")
        print("3. Place it in the 'checkpoints' directory as 'wav2lip.pth'")

if __name__ == '__main__':
    download_models() 