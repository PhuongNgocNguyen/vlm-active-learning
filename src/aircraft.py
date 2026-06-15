import os
import tarfile
import requests
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

DATA_DIR = "dataset/fgvc-aircraft"
FILENAME = "fgvc-aircraft-2013b.tar.gz"
URL = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"

def download_and_extract():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    file_path = os.path.join(DATA_DIR, FILENAME)
    extract_folder = os.path.join(DATA_DIR, "fgvc-aircraft-2013b")
    
    if os.path.exists(extract_folder):
        return

    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        print(f"Downloading FGVC-Aircraft...")
        try:
            response = requests.get(URL, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            with open(file_path, 'wb') as f, tqdm(
                desc=FILENAME, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)
        except Exception as e:
            print(f"Error downloading: {e}")
            return

    print("Extracting file...")
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)
    except Exception as e:
        print(f"Extraction error: {e}")

def read_metadata_df(root_path, split):
    variant_file = os.path.join(root_path, "data", f"images_variant_{split}.txt")
    manufacturer_file = os.path.join(root_path, "data", f"images_manufacturer_{split}.txt")
    
    with open(manufacturer_file, 'r') as f:
        manu_lines = f.readlines()
    manu_dict = {}
    for line in manu_lines:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
            manu_dict[parts[0]] = parts[1]
            
    data = []
    with open(variant_file, 'r') as f:
        variant_lines = f.readlines()
        
    for line in variant_lines:
        parts = line.strip().split(' ', 1)
        img_id = parts[0]
        variant_name = parts[1]
        manu_name = manu_dict.get(img_id, "Unknown")
        
        if manu_name not in variant_name:
            full_name = f"{manu_name} {variant_name}"
        else:
            full_name = variant_name
            
        img_path = os.path.join(root_path, "data", "images", f"{img_id}.jpg")
        
        data.append({
            'img_id': img_id,
            'filepath': img_path,
            'full_label_str': full_name,
            'weak_label_str': manu_name
        })
        
    return pd.DataFrame(data)

def load_aircraft_train_test_as_numpy(image_size=224, use_vlm_labels=True):
    download_and_extract()
    root_path = os.path.join(DATA_DIR, "fgvc-aircraft-2013b")
    
    print("Processing Metadata...")
    train_df = read_metadata_df(root_path, "trainval")
    test_df = read_metadata_df(root_path, "test")
    
    # Mapping Labels
    unique_full = sorted(train_df['full_label_str'].unique())
    full_to_idx = {name: i for i, name in enumerate(unique_full)}
    
    unique_weak = sorted(train_df['weak_label_str'].unique())
    weak_to_idx = {name: i for i, name in enumerate(unique_weak)}
    
    print(f"Stats: {len(unique_full)} Variants, {len(unique_weak)} Manufacturers")

    def load_images_from_df(df):
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)), 
            transforms.CenterCrop(image_size)
        ])
        
        image_list = []
        filepaths = df['filepath'].values
        
        for filepath in tqdm(filepaths, desc="Loading Images", leave=False, ncols=80):
            try:
                img = Image.open(filepath).convert("RGB")
                img = transform(img)
                image_list.append(np.array(img))
            except:
                image_list.append(np.zeros((image_size, image_size, 3), dtype=np.uint8))
                
        return np.array(image_list)

    # --- TRAIN DATA ---
    print("Loading Train Images...")
    train_images = load_images_from_df(train_df)
    train_labels_full = train_df['full_label_str'].map(full_to_idx).values
    train_labels_weak_true = train_df['weak_label_str'].map(weak_to_idx).values

    # --- PREDICTED WEAK LABELS ---
    predicted_weak_labels_path = "result/air_predicted_weak_labels.npy"
    
    if use_vlm_labels and os.path.exists(predicted_weak_labels_path):
        print(f"Found predicted labels at: {predicted_weak_labels_path}")
        predicted_labels_str = np.load(predicted_weak_labels_path, allow_pickle=True)
        
        if len(predicted_labels_str) != len(train_df):
            print(f"Mismatch length! Using GT instead.")
            train_labels_weak = train_labels_weak_true.copy()
        else:
            train_labels_weak = []
            for label in predicted_labels_str:
                label = str(label).strip()
                class_id = weak_to_idx.get(label, -1)  # -1 if the label is unknown.
                train_labels_weak.append(class_id)
            train_labels_weak = np.array(train_labels_weak)
            print(f"Loaded {len(train_labels_weak)} predicted weak labels.")
    elif not use_vlm_labels:
        print("VLM weak labels disabled. Using Ground Truth as weak labels.")
        train_labels_weak = train_labels_weak_true.copy()
    else:
        print("No predicted file found. Using Ground Truth as weak labels.")
        train_labels_weak = train_labels_weak_true.copy()

    # --- TEST DATA ---
    print("Loading Test Images...")
    test_images = load_images_from_df(test_df)
    test_labels_full = test_df['full_label_str'].map(full_to_idx).values
    test_labels_weak = test_df['weak_label_str'].map(weak_to_idx).values

    return (train_images, train_labels_full, train_labels_weak_true, train_labels_weak, 
            test_images, test_labels_full, test_labels_weak)


def initial_aircraft(labels_full, k=3, seed=42, save_dir="result", index_filename="round0_full_air.npy"):
    """
    Select k initial full-label samples per class and save their indices.
    This function does not modify images or labels.
    """
    os.makedirs(save_dir, exist_ok=True)
    index_file_path = os.path.join(save_dir, index_filename)
    
    # 1. Load and return existing indices if the file already exists.
    if os.path.exists(index_file_path):
        print(f"Founded round0 index: '{index_file_path}'")
        return np.load(index_file_path)

    # 2. Otherwise, create a new balanced initial index set.
    print(f"File not found. Creating a new index ({k} images per class)...")
    np.random.seed(seed)
    
    subset_indices = []
    unique_classes = np.unique(labels_full)  # List all classes.
    
    for cls_idx in unique_classes:
        # Collect all sample indices for this class.
        indices = np.where(labels_full == cls_idx)[0]
        
        # Randomly choose k samples.
        if len(indices) >= k:
            selected = np.random.choice(indices, k, replace=False)
        else:
            selected = indices  # Use all samples if fewer than k are available.
            
        subset_indices.append(selected)
    
    # Merge selected class-wise indices into one array.
    final_indices = np.concatenate(subset_indices, axis=0)
    
    # 3. Save indices.
    np.save(index_file_path, final_indices)
    print(f"Index saved to: {index_file_path}")
    
    return final_indices

if __name__ == "__main__":
    download_and_extract()
    train_images, train_labels_full, train_labels_weak_true, train_labels_weak, test_images, test_labels_full, test_labels_weak = load_aircraft_train_test_as_numpy()
