import argparse
from pathlib import Path
from utils import set_device

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--model_backbone', default='vit_b_16', type=str, 
                        choices=['resnet18', 'resnet34', 'resnet50', 
                                 'efficientnetv2', 'vit_b_16', 'convnext_b'])
    parser.add_argument('--optimizer', default='adam', type=str, choices=['sgd','adam'])
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--output_dir', default='result/', type=str)

    parser.add_argument('--dataset', default='cub200', type=str, choices=['cub200', 'aircraft'])
    parser.add_argument('--uncertainty', default='margin', type=str,  choices=['ent', 'max_conf', 'margin'])
    parser.add_argument('--num_rounds', default=6, type=int)
    parser.add_argument('--budget', default=150, type=int)
    parser.add_argument('--cost_weak', default=0.02, type=float) # cost_full=1.0
    parser.add_argument('--no_transition_matrix', action='store_true',
                        help='Disable forward correction and train weak labels with cross entropy.')
    parser.add_argument('--no_vlm_labels', action='store_true',
                        help='Ignore Gemini/VLM weak-label files and use ground-truth weak labels.')

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.device = set_device(args)

    if args.dataset =='cub200':
        args.num_classes, args.num_super_classes = 200, 70
        args.image_size = 224
    elif args.dataset == 'aircraft':
        args.num_classes, args.num_super_classes = 100, 30
        args.image_size = 224

    return args
