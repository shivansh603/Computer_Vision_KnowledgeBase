import os
import requests
import argparse

# Install pipreqs if not already installed and generate requirements.txt
def setup():
    os.system("pip3 install pipreqs")
    os.system("pipreqs . --force --ignore=tests")
os.system("pip3 install -r requirements.txt")




def get_model_id(tags):
    """
    Fetch the model ID from the AutoAI backend using specified tags.
    
    Parameters:
    tags (str): Tags to filter the model.
    
    Returns:
    str: The model ID.
    """
    url = 'https://autoai-backend-exjsxe2nda-uc.a.run.app/model'
    payload = {'tags': tags}
    response = requests.get(url, params=payload, headers={'accept': 'application/json'})
    
    if response.status_code == 200:
        return response.json().get("-id")
    else:
        print(f"Failed to fetch model ID. Status code: {response.status_code}")
        return None

    
    
def get_collections(model_id):
    """
    Retrieve the collections associated with a given model ID.
    
    Parameters:
    model_id (str): The model ID.
    
    Returns:
    dict: Collection data.
    """
    url = 'https://autoai-backend-exjsxe2nda-uc.a.run.app/collection/dataSet'
    payload = {'model': model_id}
    headers = {'accept': 'application/json'}
    
    response = requests.get(url, params=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch collections. Status code: {response.status_code}")
        return None


    
def get_image_ids(collection_id):
    """
    Get image IDs from a specified collection ID.
    
    Parameters:
    collection_id (str): The collection ID.
    
    Returns:
    list: List of image IDs.
    """
    url = f'https://autoai-backend-exjsxe2nda-uc.a.run.app/collection/dataSet/{collection_id}'
    headers = {'accept': 'application/json'}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json().get("resources", [])
    else:
        print(f"Failed to fetch image IDs. Status code: {response.status_code}")
        return []

    
    
def get_image_signed_url(image_id):
    """
    Retrieve the signed URL for an image given its ID.
    
    Parameters:
    image_id (str): The image ID.
    
    Returns:
    str: The signed URL for the image.
    """
    url = f'https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/{image_id}'
    headers = {'accept': 'application/json'}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json().get("resource")
    else:
        print(f"Failed to fetch signed URL for image ID {image_id}. Status code: {response.status_code}")
        return None

    
    
    
def download_images(collection_id):
    """
    Download images from a specified collection and save them locally.
    
    Parameters:
    collection_id (str): The collection ID.
    """
    if not os.path.exists("sample_images"):
        os.mkdir("sample_images")
    
    image_ids = get_image_ids(collection_id)
    
    print("Image IDs:")
    print(image_ids)
    
    for image_id in image_ids:
        image_url = get_image_signed_url(image_id)
        
        if image_url:
            image_response = requests.get(image_url)
            
            if image_response.status_code == 200:
                with open(f"sample_images/{image_id}.png", "wb") as f:
                    f.write(image_response.content)
            else:
                print(f"Failed to download image ID {image_id}. Status code: {image_response.status_code}")
                
        
        
    print("Images downloaded successfully")
    os.system("zip -r sample_images.zip sample_images/*")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images from a dataset collection")
    parser.add_argument("--collection_id", type=str, required=True, help="Collection ID of the dataset")
    args = parser.parse_args()
    
    setup()
    download_images(args.collection_id)
