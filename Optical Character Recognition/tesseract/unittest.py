# import unittest
import os 
# collection_id = "6567042d028055e259e9c70c"
# os.system(f"python3 setup.py --collection_id '{collection_id}'")
from inference import extract_text




if __name__ == '__main__':
    text = extract_text("temp.png")

    print(text)

