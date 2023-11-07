import argparse
import re
import sys
import os
sys.path.append(os.path.dirname(__file__))
from codes import *

def get_ifeature(seq, type):
    fastas = [['test', seq]]
    # kw = {'path': args.filePath, 'train': args.trainFile, 'label': args.labelFile, 'order': myOrder}
    kw = {'order': 'ACDEFGHIKLMNPQRSTVWY'}
    myFun = type + '.' + type + '(fastas, **kw)'
    encodings = eval(myFun)
    return encodings[1][1:]