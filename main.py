# =============================================================================
# Import required libraries
# =============================================================================
import argparse
import numpy as np

import torch
from torch import nn

from datasets import make_data_loader
from utils import *
from models import Annotator
from loss_functions import MultiLabelLoss
from engine import Engine


# =============================================================================
# Define hyperparameters
# =============================================================================
parser = argparse.ArgumentParser(
    description='PyTorch Training for Automatic Image Annotation')
parser.add_argument('--seed', default=20, type=int,
                    help='seed for initializing training')
parser.add_argument('--data_root_dir', default='./datasets/', type=str)
parser.add_argument('--image-size', default=448, type=int)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--num_workers', default=2, type=int,
                    help='number of data loading workers (default: 2)')
parser.add_argument('--learning-rate', default=0.0001, type=float)
parser.add_argument('--loss-function', metavar='NAME',
                    help='loss function (e.g. BCELoss)')
parser.add_argument('--data', metavar='NAME',
                    help='dataset name (e.g. Corel-5k)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluation of the model on the validation set')
parser.add_argument('--mixup', dest='mixup', action='store_true')
parser.add_argument(
    '--save_dir', default='./checkpoints/', type=str, help='save path')


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    is_train = True if not args.evaluate else False

    train_loader, validation_loader, classes = make_data_loader(args)

    model = Annotator(args=args,
                      num_classes=len(classes),
                      hidden_size=1024,
                      num_heads=8,
                      num_decoder_layers=1,
                      feedforward_size=2048,
                      norm_first=True,
                      remove_self_attn=True,
                      keep_query_position=True)

    if args.loss_function == 'BCELoss':
        criterion = nn.MultiLabelSoftMarginLoss()
    elif args.loss_function == 'FocalLoss':
        criterion = MultiLabelLoss(gamma_neg=3,
                                   gamma_pos=3,
                                   pos_margin=0,
                                   neg_margin=0)
    elif args.loss_function == 'AsymmetricLoss':
        criterion = MultiLabelLoss(gamma_neg=4,
                                   gamma_pos=0,
                                   pos_margin=0,
                                   neg_margin=0.05)
    elif args.loss_function == 'proposedLoss':
        criterion = MultiLabelLoss(gamma_neg=4,
                                   gamma_pos=3,
                                   pos_margin=1.1,
                                   neg_margin=0.05,
                                   threshold=0.25)

    engine = Engine(args,
                    model,
                    criterion,
                    train_loader,
                    validation_loader,
                    classes)

    if is_train:
        engine.initialization()
        engine.train_iteration()
    else:
        engine.initialization()
        engine.load_model()
        engine.validation(dataloader=validation_loader)
        # show images and predicted labels
        loader = iter(validation_loader)
        images, annotations = next(loader)
        predicted_batch_plot(args,
                             classes,
                             model,
                             images,
                             annotations)
        #
        visualize_att(args,
                      classes,
                      model,
                      images,
                      annotations)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
