import argparse
import os
import numpy as np
import tensorflow as tf
from dncnn import DnCNN
from utils import load_images_from_folder

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--ndct_training_data', dest='ndct_training_data', default='/mmfs1/gscratch/uwb/CT_images/train/images', help='ndct dataset for training')
parser.add_argument('--ldct_training_data', dest='ldct_training_data', default='/mmfs1/gscratch/uwb/CT_images/train/images', help='ldct dataset for training')
parser.add_argument('--eval_data', dest='eval_data', default='/mmfs1/gscratch/uwb/CT_images/train/images', help='evaluation dataset')
parser.add_argument('--test_dir', dest='test_dir', default='/mmfs1/gscratch/uwb/CT_images/train/images', help='test output directory')
args = parser.parse_args()

def denoiser_train(denoiser, lr):
    ndct_data = load_images_from_folder(args.ndct_training_data)
    ldct_data = load_images_from_folder(args.ldct_training_data)
    eval_files = sorted([f for f in os.listdir(args.eval_data) if f.endswith('.png')])
    eval_data = load_images_from_folder(args.eval_data)
    ldct_eval_data = load_images_from_folder(args.eval_data)  # Assuming evaluation data is the same as eval_data

    denoiser.train(ndct_data, ldct_data, eval_data, ldct_eval_data, batch_size=args.batch_size, ckpt_dir=args.ckpt_dir, epoch=args.epoch, lr=lr,
                   sample_dir=args.sample_dir)

def denoiser_test(denoiser, test_data, ckpt_dir, save_dir):
    denoiser.test(test_data, ckpt_dir=ckpt_dir, save_dir=save_dir)

def main():
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    lr = args.lr * np.ones([args.epoch])
    lr[30:] = lr[0] / 10.0

    if args.use_gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)
        
        # Using TensorFlow v2 session management
        with tf.compat.v1.Session() as sess:
            model = DnCNN(sess)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                # Implement your test logic here
                test_data = load_images_from_folder(args.test_dir)
                denoiser_test(model, test_data, args.ckpt_dir, args.test_dir)
            else:
                print('[!]Unknown phase')
    else:
        # Using TensorFlow v1 session management
        with tf.compat.v1.Session() as sess:
            model = DnCNN(sess, batch_size=args.batch_size)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                # Implement your test logic here
                test_data = load_images_from_folder(args.test_dir)
                denoiser_test(model, test_data, args.ckpt_dir, args.test_dir)
            else:
                print('[!]Unknown phase')

if __name__ == '__main__':
    main()

