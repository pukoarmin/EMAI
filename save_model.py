import sys, time, os
import getopt
import argparse
import shutil
import tensorflow as tf

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
    tf.saved_model.save(model, direct)
