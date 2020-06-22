import os
import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='pupil')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument("--data", type=str, default="illumination",
                      help="center (w/o illumination) or illumination (w/ illumination)")
data_arg.add_argument('--batch_size', type=int, default=8,
                      help='# of images in each batch of data')

train_arg = add_argument_group('Training Params')
train_arg.add_argument('--epochs', type=int, default=30,
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=1e-3,
                       help='Initial learning rate value')
train_arg.add_argument('--random_seed', type=int, default=0,
                       help='Random seed')
train_arg.add_argument("--noise_std", type=float, default=0.005,
                       help="Noise standard deviation")
train_arg.add_argument("--is_pupil_train", type=int, default=1,
                       help="0 (fix pupil) or 1 (train pupil)")
train_arg.add_argument("--is_illu", type=int, default=1,
                       help="0 (w/o illumination) or 1 (w/ illumination)")

misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--gpu_number', type=int, default=0,
                      help="Which GPU to use")


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed