# -*- coding:utf-8 -*-
import argparse
import sys
from data.voc_generator import PascalVocGenerator
import tensorflow as tf
import cv2
from model import resnet, fpn, loss, backbone
from model.retinanet import default_submodels
from utils.config import make_training_config

def check_args(args):
    assert args.data_dir is not None, "No input argument: --data_dir."
    assert args.set_type is not None, "No input argument: --set_type."
    return args

def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument("--user", help="Who runs the training script.", default="luojh")
    parser.add_argument("--data_dir", help="Date path.", default=None)
    parser.add_argument("--set_type", help="Type of data set.", default=None)
    parser.add_argument("--train_config", help="Path to training config file.", default="./config/train.yaml")
    return check_args(parser.parse_args(args))

def get_callbacks():
    pass

def main(args = None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    print ("User:{}, Set_type:{}".format(args.user, args.set_type))

    config = make_training_config(args)

    generator = PascalVocGenerator(args.data_dir, "train")
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        training_backbone = backbone("resnet50")
        model = training_backbone.retinanet(num_classes=20)
    model.compile(
        loss={
            'regression'    : loss.smooth_l1(),
            'classification': loss.focal()
        },
        optimizer=tf.keras.optimizers.Adam(lr=1e-5, clipnorm=0.001)
    )

    print (model.summary())
    model.fit(
        generator, 
        steps_per_epoch=10, 
        epochs=2,
        workers=1, 
        use_multiprocessing=1)

if __name__ == "__main__":
    main()
