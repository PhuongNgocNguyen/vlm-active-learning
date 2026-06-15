import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score
import copy
import torch.nn as nn

from model import set_optimizer, forward_loss
from loader import set_train_loader, set_test_loader
from utils import AverageMeter

def train(
    X_train,
    y_train,
    annotation_type,
    val_loader,
    model,
    num_classes, 
    num_super_classes,
    optimizer,
    lr,
    image_size,
    batch_size,
    num_workers,
    num_epochs,
    device,
    transition_matrix=None, 
    seed=42
):    
    
    # -------
    history_train_weak_acc = []
    history_val_weak_acc = []
    history_train_full_acc = []
    history_val_full_acc = []
    
    T_tensor = None
    if transition_matrix is not None :
        print("\n[Train] Use Forward Correction")
        T_tensor = torch.from_numpy(transition_matrix).float().to(device)
    else:
        print("\n[Train] T_tensor is None. Use Cross-Entropy")

    # ----- training by weak supervision --------------

    optimizer = set_optimizer(model=model, optimizer=optimizer, lr=lr)
    
    criterion_weak = nn.CrossEntropyLoss(ignore_index=-1, reduction='none').to(device)
    load_index = np.arange(len(X_train))[annotation_type != 0]

    train_loader = set_train_loader(X_train, y_train, load_index, 
                                    annotation_type=annotation_type,
                                    image_size=image_size, batch_size=batch_size, 
                                    num_workers=num_workers, 
                                    seed=seed)
    
    # -----
    history_train_loss, history_train_acc_full = [], []
    best_val_acc = 0.0
    train_loss, val_loss = AverageMeter(), AverageMeter()
    
    for epoch in tqdm(range(num_epochs), leave=False, ncols=50):

        # ---- Train ----
        model.train()
        pred, gt = [], []

        for batch in train_loader:
            X = batch['X'].to(device, dtype=torch.float32)
            y_weak = batch['y_weak'].to(device, dtype=torch.long)
            n = X.size(0)
            types = batch['annotation_type'].to(device, dtype=torch.long)
            
            _, logits_weak, _ = model(X) 
            pixel_losses = torch.zeros(n, device=device)
            
            mask_1 = (types == 1)
            mask_2 = (types == 2)
            
            if mask_1.sum() > 0:
                loss_noisy, _ = forward_loss(logits_weak[mask_1], y_weak[mask_1], T_tensor, reduction='none')
                pixel_losses[mask_1] = loss_noisy
                
            if mask_2.sum() > 0:
                pixel_losses[mask_2] = criterion_weak(logits_weak[mask_2], y_weak[mask_2])
                    
            valid_sample_count = (y_weak != -1).sum()
            
            if valid_sample_count > 0:
                loss = pixel_losses.sum() / valid_sample_count
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss.update(loss.item(), n=n)
            pred.extend(logits_weak.argmax(dim=-1).detach().cpu().numpy())
            gt.extend(y_weak.detach().cpu().numpy())

        history_train_loss.append(train_loss.avg)
        gt, pred = np.array(gt), np.array(pred)
        train_acc = accuracy_score(gt, pred)
        history_train_acc_full.append(train_acc*100)

        train_loss.reset()

        # ---- Validation ----
        model.eval()
        pred, gt = [], []
        with torch.no_grad():
            for batch in val_loader:
                X = batch['X'].to(device, dtype=torch.float32)
                y_weak = batch['y_weak'].to(device, dtype=torch.long)
                n = X.size(0)

                _, logits_weak, _ = model(X)
                loss = criterion_weak(logits_weak, y_weak).mean()

                val_loss.update(loss.item(), n=n)
                pred.extend(logits_weak.argmax(dim=-1).detach().cpu().numpy())
                gt.extend(y_weak.detach().cpu().numpy())

        gt, pred = np.array(gt), np.array(pred)
        val_acc = accuracy_score(gt, pred)

        if val_acc*100 >= best_val_acc:
            best_val_acc = val_acc*100
            best_model_param = copy.deepcopy(model.state_dict())

        val_loss.reset()

        # -------
        history_train_weak_acc.append(train_acc*100)
        history_val_weak_acc.append(val_acc*100)
    
    model.load_state_dict(best_model_param)


    # ---- training by full supervision ----

    optimizer = set_optimizer(model=model, optimizer=optimizer, lr=lr)
    load_index = np.arange(len(X_train))[annotation_type == 2]

    train_loader = set_train_loader(X_train, y_train, load_index, 
                                    annotation_type=annotation_type, 
                                    image_size=image_size, batch_size=batch_size, 
                                    num_workers=num_workers, 
                                    seed=seed)

    criterion_full = nn.CrossEntropyLoss(ignore_index=-1).to(device)
        
    # ----
    history_train_loss, history_train_acc_full = [], []
    best_val_acc = 0.0
    train_loss, val_loss = AverageMeter(), AverageMeter()
    
    for epoch in tqdm(range(num_epochs), leave=False, ncols=50):

        # ---- Train ----
        model.train()
        pred, gt = [], []
        
        for batch in train_loader:
            X = batch['X'].to(device, dtype=torch.float32)
            y_full = batch['y_full'].to(device, dtype=torch.long)
            y_weak = batch['y_weak'].to(device, dtype=torch.long)
            n = X.size(0)

            _, logits_weak, logits_full = model(X)

            loss = criterion_full(logits_full, y_full)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss.update(loss.item(), n=n)
            pred.extend(logits_full.argmax(dim=-1).detach().cpu().numpy())
            gt.extend(y_full.detach().cpu().numpy())

        history_train_loss.append(train_loss.avg)
        gt, pred = np.array(gt), np.array(pred)
        train_acc = accuracy_score(gt, pred)
        history_train_acc_full.append(train_acc*100)
        
        train_loss.reset()

        # ---- Validation ----
        model.eval()
        pred, gt = [], []
        with torch.no_grad():
            for batch in val_loader:
                X = batch['X'].to(device, dtype=torch.float32)
                y_full = batch['y_full'].to(device, dtype=torch.long)
                n = X.size(0)

                _, _, logits_full = model(X)
                loss = criterion_full(logits_full, y_full)

                val_loss.update(loss.item(), n=n)
                pred.extend(logits_full.argmax(dim=-1).detach().cpu().numpy())
                gt.extend(y_full.detach().cpu().numpy())

        gt, pred = np.array(gt), np.array(pred)
        val_acc = accuracy_score(gt, pred)

        if val_acc*100 >= best_val_acc:
            best_val_acc = val_acc*100
            best_model_param = copy.deepcopy(model.state_dict())

        val_loss.reset()
        # -------
        history_train_full_acc.append(train_acc*100)
        history_val_full_acc.append(val_acc*100)


    model.load_state_dict(best_model_param)

    # -------
    history_all = {
    "train_weak_acc": history_train_weak_acc,
    "val_weak_acc": history_val_weak_acc,
    "train_full_acc": history_train_full_acc,
    "val_full_acc": history_val_full_acc
    }

    return model, best_val_acc, history_all

def test(
    model,
    test_loader,
    device
):
    
    model.eval()
    pred, gt = [], []
    with torch.no_grad():
        for batch in test_loader:
            X = batch['X'].to(device, dtype=torch.float32)
            y_full = batch['y_full'].to(device, dtype=torch.long)

            _, _, logits_full = model(X)
            pred.extend(logits_full.argmax(dim=-1).detach().cpu().numpy())
            gt.extend(y_full.detach().cpu().numpy())

    gt, pred = np.array(gt), np.array(pred)
    test_acc = accuracy_score(gt, pred)*100
    
    return test_acc