import torch
import numpy as np
from torchvision import transforms
from utils import seed_worker
from cub200 import load_cub_2011_train_test_as_numpy
from aircraft import load_aircraft_train_test_as_numpy


def split_val_per_class(y, n_val_per_class=2, seed=42):
    rng = np.random.default_rng(seed)
    idx_val = []
    idx_train = []

    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)

        if len(idx_c) < n_val_per_class:
            raise ValueError(
                f"Class {c} has only {len(idx_c)} samples, "
                f"but {n_val_per_class} are required for validation."
            )

        idx_val.extend(idx_c[:n_val_per_class])
        idx_train.extend(idx_c[n_val_per_class:])
    
    assert len(set(idx_train) & set(idx_val)) == 0
    return np.array(idx_train), np.array(idx_val)


def download_dataset(dataset='cub200', use_vlm_labels=True):
    # Initialize variables to avoid UnboundLocalError if branching logic changes.
    y_train_full, y_val_full = None, None
    y_train_weak_gt, y_val_weak_gt = None, None
    y_train_weak_pred, y_val_weak_pred = None, None

    if dataset == 'cub200':
        X_train, y_train, y_train_weak_gt, y_train_weak_pred, X_test, y_test, y_test_weak = load_cub_2011_train_test_as_numpy(
            use_vlm_labels=use_vlm_labels)
        
        idx_train, idx_val = split_val_per_class(y_train, n_val_per_class=2, seed=42)

        X_train, X_val = X_train[idx_train], X_train[idx_val]
        y_train_full, y_val_full = y_train[idx_train], y_train[idx_val]
        y_train_weak_gt, y_val_weak_gt = y_train_weak_gt[idx_train], y_train_weak_gt[idx_val]
        y_train_weak_pred, y_val_weak_pred = y_train_weak_pred[idx_train], y_train_weak_pred[idx_val]

    elif dataset == 'aircraft':
        X_train, y_train, y_train_weak_gt, y_train_weak_pred, X_test, y_test, y_test_weak = load_aircraft_train_test_as_numpy(
            use_vlm_labels=use_vlm_labels)
        
        idx_train, idx_val = split_val_per_class(y_train, n_val_per_class=2, seed=42)
    
        X_train, X_val = X_train[idx_train], X_train[idx_val]
        y_train_full, y_val_full = y_train[idx_train], y_train[idx_val]
        y_train_weak_gt, y_val_weak_gt = y_train_weak_gt[idx_train], y_train_weak_gt[idx_val]
        y_train_weak_pred, y_val_weak_pred = y_train_weak_pred[idx_train], y_train_weak_pred[idx_val]
        
    else:
        raise NameError("dataset {} is not supported".format(dataset))

    y_train = {'full': y_train_full, 'weak_gt': y_train_weak_gt, 'weak_pred': y_train_weak_pred}
    y_val = {'full': y_val_full, 'weak_gt': y_val_weak_gt, 'weak_pred': y_val_weak_pred}
    y_test = {'full': y_test, 'weak': y_test_weak}

    return X_train, y_train, X_val, y_val, X_test, y_test

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, load_index, transform, annotation_type=None):
        self.X = X
        self.y = y
        self.load_index = load_index
        self.transform = transform
        self.annotation_type = annotation_type

    def __len__(self):
        return len(self.load_index)

    def __getitem__(self, idx):
            original_idx = self.load_index[idx]

            X = self.X[original_idx]
            X = self.transform(X)
            
            y_full = self.y['full'][original_idx]
            
            if 'weak_gt' in self.y:
                y_weak_gt = self.y['weak_gt'][original_idx]

                current_type = -1
                
                if self.annotation_type is not None:
                    y_weak_pred = self.y['weak_pred'][original_idx]
                    current_type = self.annotation_type[original_idx]
                    
                    if current_type == 1:      # annotation_type = 1
                            selected_y_weak = y_weak_pred 
                            
                    elif current_type == 2:    # annotation_type = 2
                        selected_y_weak = y_weak_gt  # Ground Truth
                        
                    else:
                        selected_y_weak = -1
                        raise ValueError(f"Unexpected annotation_type: {current_type}. Only 1 or 2 are expected")
                
                # VALIDATION
                else:
                    selected_y_weak = y_weak_gt
                
                return {'X': X, 'y_full': y_full, 'y_weak': selected_y_weak, 'idx': original_idx, 'annotation_type': current_type}

            else:
                # TEST LOADER
                y_weak = self.y['weak'][original_idx]
                return {'X': X, 'y_full': y_full, 'y_weak': y_weak, 'idx': original_idx, 'annotation_type': -1}
    
def set_train_loader(X, y, load_index, annotation_type, image_size, batch_size=128, num_workers=4,seed=42):
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=image_size//8),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_dataset = Dataset(X=X, y=y, load_index=load_index, 
                            transform=train_transforms, 
                            annotation_type=annotation_type,
                            )
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, worker_init_fn=seed_worker,generator=g)
    return train_loader

def set_test_loader(X, y, image_size, batch_size=128, num_workers=4):
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_dataset = Dataset(X=X, y=y, load_index=np.arange(len(X)), transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,  num_workers=num_workers, pin_memory=False)
    return test_loader

def set_unlabeled_loader(X, y, load_index, image_size, batch_size=128, num_workers=4):
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        
    ])
    unlabeled_dataset = Dataset(X=X, y=y, load_index=load_index, transform=test_transforms)
    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return unlabeled_loader
        
if __name__ == '__main__':    
    X_train, y_train, X_val, y_val, X_test, y_test = download_dataset(dataset='aircraft')
