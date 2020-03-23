# -*- coding:utf-8 -*-
import argparse
import sys
import os
from data.voc_generator import PascalVocGenerator
import tensorflow as tf
import cv2
from model import resnet, fpn, loss, backbone
from model.retinanet import default_submodels, retinanet_bbox
from utils.config import make_training_config
from callback import callbacks

def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument("--train_config", help="Path to training config file.", default="./config/train.yaml")
    return parser.parse_args(args)

def creat_models():
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        training_backbone = backbone("resnet50")
        training_model = training_backbone.retinanet(num_classes=20)
    predict_model = retinanet_bbox(training_model)
    return training_model, predict_model

def get_callbacks(logdir, predict_model, val_generator, save_dir="./save_model"):
    custom_callbacks = callbacks.MyCustomCallback(logdir, predict_model, val_generator)
    checkpoint_path = os.path.join(save_dir, "{epoch:02d}-{mAP:.2f}.ckpt")
    save_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1, monitor="mAP", save_best_only=True)
    return [custom_callbacks, save_callback]

def main(args = None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    print ("Using config file: {}".format(args.train_config))

    training_model, predict_model = creat_models()
    training_generator = PascalVocGenerator("/home/luo13/workspace/datasets/detection/voc2012/VOCdevkit/VOC2012", "train", batch_size=2)
    print ("Training batchs per epoch: {}".format(training_generator.__len__()))
    batchs_per_epoch = training_generator.__len__()

    val_generator = PascalVocGenerator("/home/luo13/workspace/datasets/detection/voc2012/VOCdevkit/VOC2012", "val", batch_size=2)
    print ("Valuating batchs per epoch: {}".format(val_generator.__len__()))

    logdir = tf.summary.create_file_writer("./logs")
    
    custom_callbacks = get_callbacks(logdir, predict_model, val_generator)

    training_model.compile(
        loss={
            'bbox_regression'    : loss.smooth_l1(),
            'classification': loss.focal()
        },
        optimizer=tf.keras.optimizers.Adam(lr=1e-5, clipnorm=0.001)
    )

    # print (training_model.summary())
    training_model.fit(
        training_generator, 
        steps_per_epoch=batchs_per_epoch, 
        epochs=10,
        workers=1, 
        callbacks=custom_callbacks,
        use_multiprocessing=False)

if __name__ == "__main__":
    main()
