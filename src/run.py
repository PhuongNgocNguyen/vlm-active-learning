import sys
import numpy as np
import os
from arguments import parse_option
from ISO import query
from model import set_model
from loader import download_dataset, set_test_loader
from train_nn import train, test
from utils import set_reproductibility, Logger
from aircraft import initial_aircraft
from cub200 import initial_cub

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def dataset_artifact_suffix(dataset):
    if dataset == 'cub200':
        return 'cub'
    if dataset == 'aircraft':
        return 'air'
    raise ValueError(f"Unsupported dataset: {dataset}")

def main():
    
    args = parse_option()
    sys.stdout = Logger(file_path=f'{args.output_dir}/log.txt')
    print(f'argument: {args}')

    history_dir = os.path.join(args.output_dir, "history")
    os.makedirs(history_dir, exist_ok=True)
    use_transition_matrix = not args.no_transition_matrix
    use_vlm_labels = not args.no_vlm_labels

    set_reproductibility(seed=args.seed)

    print(f'dataset: {args.dataset}')
    X_train, y_train, X_val, y_val, X_test, y_test = download_dataset(
        dataset=args.dataset,
        use_vlm_labels=use_vlm_labels)

    artifact_suffix = dataset_artifact_suffix(args.dataset)
    round0_full_path = os.path.join(args.output_dir, f"round0_full_{artifact_suffix}.npy")
    transition_matrix_path = os.path.join(args.output_dir, f"transition_matrix_round0_{artifact_suffix}.npy")
    print(f'use transition matrix: {use_transition_matrix}')
    print(f'use VLM weak labels: {use_vlm_labels}')
    
    if args.dataset == 'cub200':
        print("Checking initial file for CUB200...")
        initial_cub(labels_full=y_train['full'], k=3, seed=17,
                    save_dir=args.output_dir,
                    index_filename=os.path.basename(round0_full_path))
    elif args.dataset == 'aircraft':
        print("Checking initial file for Aircraft...")
        initial_aircraft(labels_full=y_train['full'], k=3, seed=17,
                         save_dir=args.output_dir,
                         index_filename=os.path.basename(round0_full_path))
        
    val_loader = set_test_loader(X=X_val, y=y_val, 
                                 image_size=args.image_size, 
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers)
    test_loader = set_test_loader(X=X_test, y=y_test, 
                                 image_size=args.image_size, 
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers)
    
    model = set_model(model_backbone=args.model_backbone, 
                      num_classes=args.num_classes, 
                      num_super_classes=args.num_super_classes).to(args.device)
    
    annotation_type = np.zeros(len(X_train), dtype=int)  # 0: unlabeled, 1: weak label, 2: full label

    for round in range(args.num_rounds):
        print('----------------------------------------------')
        print(f'Round: {round+1}/{args.num_rounds}')
        
        # ---- annotation ----
        print('query...')
        annotation_type = query(round=round,
                                budget=args.budget,
                                cost_weak=args.cost_weak,
                                X_train=X_train,
                                y_train=y_train,
                                annotation_type=annotation_type,
                                val_loader=val_loader,
                                model=model,
                                num_classes=args.num_classes, 
                                num_super_classes=args.num_super_classes,
                                optimizer=args.optimizer,
                                lr=args.lr,
                                image_size=args.image_size, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers, 
                                num_epochs=args.num_epochs,
                                uncertainty=args.uncertainty,
                                device=args.device,
                                round0_full_path=round0_full_path,
                                transition_matrix_path=transition_matrix_path,
                                use_transition_matrix=use_transition_matrix,
                                seed=args.seed)
    
        print('number of weak supervision: %d/%d (%.2f%%)'%(
            sum(annotation_type==1), len(X_train), 
            (sum(annotation_type==1)/len(X_train))*100))
        
        print('number of full supervision: %d/%d (%.2f%%)'%(
            sum(annotation_type==2), len(X_train), 
            (sum(annotation_type==2)/len(X_train))*100))
        
        T_tensor = None
        T_path = transition_matrix_path
        
        if use_transition_matrix and os.path.exists(T_path):
            try:
                print(f"Loading transition matrix from {T_path}...")
                T_tensor = np.load(T_path)
                print(f"Loaded transition matrix for Round {round+1}, shape:", T_tensor.shape)
            except Exception as e:
                print(f"Error loading matrix: {e}")
                T_tensor = None
        elif not use_transition_matrix:
            print("Transition matrix disabled.")
        else:
            print(f"File {T_path} not found!")

        # ---- training ----
        print('training...')
        
        model = model.init().to(args.device)
        model, val_acc, history = train(X_train=X_train,
                                                y_train=y_train,
                                                annotation_type=annotation_type,
                                                val_loader=val_loader,
                                                model=model,
                                                num_classes=args.num_classes, 
                                                num_super_classes=args.num_super_classes,
                                                optimizer=args.optimizer,
                                                lr=args.lr,
                                                image_size=args.image_size,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_workers,
                                                num_epochs=args.num_epochs,
                                                device=args.device,
                                                transition_matrix=T_tensor,
                                                seed=args.seed)
        
        np.savez(os.path.join(history_dir, f"round_{round+1}_history.npz"),
         train_weak_acc = history["train_weak_acc"],
         val_weak_acc = history["val_weak_acc"],
         train_full_acc = history["train_full_acc"],
         val_full_acc = history["val_full_acc"])
        
        # ---- test ----
        test_acc = test(test_loader=test_loader,
                        model=model,
                        device=args.device)
        print(f'test accuracy: {test_acc:.2f} %')

    print('----------------------------------------------')


if __name__ == '__main__':
    main()
