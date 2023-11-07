import sys, time, os
import getopt
import argparse
import shutil
import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize

from Logic import Architecture


if __name__ == "__main__":
    opts, ars = getopt.getopt(sys.argv[1:], "d:h:w:m:", ["directory=", "height=", "width=", "model=", "help"])

    model = "SweepNet"
    	
    for opt, arg in opts:
        if opt in ("-d", "--directory"):
            direct = arg
        elif opt in ("-h", "--height"):
            height = arg
        elif opt in ("-w", "--width"):
            width = arg
        elif opt in ("-m", "--model"):
            model = arg
        elif opt in ("--help"):
            help()
	
	
    model = Architecture.SweepNet(int(height), int(width))
    #tf.saved_model.save(model, direct)

    q_ds = tf.keras.preprocessing.image_dataset_from_directory(
                "/home/s3052907/EMAI/EMAI/DATASETS/D1/train/images",
                label_mode='categorical',
                validation_split=0.5,
                subset="training",
                color_mode="grayscale",
                image_size=(int(height), int(width)),
                seed=123,
                batch_size = 10)
    ev_ds = tf.keras.preprocessing.image_dataset_from_directory(
                "/home/s3052907/EMAI/EMAI/DATASETS/D1/test/images",
                label_mode='categorical',
                validation_split=0.5,
                subset="validation",
                color_mode="grayscale",
                image_size=(int(height), int(width)),
                seed=123,
                batch_size = 10)

    #quantize model
    quantizer = vitis_quantize.VitisQuantizer(model)
    quantized_model = quantizer.quantize_model(calib_dataset=q_ds, calib_step=100, calib_batch_size=10) 
    quantized_model.save('quantized_model.h5')
    quantized_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics= keras.metrics.SparseTopKCategoricalAccuracy())
    quantized_model.evaluate(ev_ds)
