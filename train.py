# -*- coding:utf-8 -*-
import argparse
import sys
from data.voc_generator import PascalVocGenerator
import tensorflow as tf
import cv2
from model import resnet

def check_args(args):
    assert args.data_dir is not None, "No input argument: --data_dir."
    assert args.set_type is not None, "No input argument: --set_type."
    return args

def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument("--user", help="Who runs the training script.", default="luojh")
    parser.add_argument("--data_dir", help="Date path.", default=None)
    parser.add_argument("--set_type", help="Type of data set.", default=None)
    return check_args(parser.parse_args(args))

def main(args = None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    print ("User:{}, Set_type:{}".format(args.user, args.set_type))

    generator = PascalVocGenerator(args.data_dir, "train")
    print (generator.num_classes())
    print (generator.size())

    # resnet50 = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=10)
    # for layer in resnet50.layers:
    #     print (layer.name)
    model = resnet.resnet50(10)
    print (model.summary())
    print (model.output)

if __name__ == "__main__":
    main()
