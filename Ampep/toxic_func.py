import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
from service import Utils as Utils
from service import GetFeature as GetFeature
from service import Trainer as Trainer
from iFeature.ifeature_func import get_ifeature


def toxic_feature(seq):
    base = os.path.dirname(__file__)
    # fastaPath = os.path.join(base, 'test.fasta')
    feature = 'AAC'
    modelPath = os.path.join(base, 'model/RandomForestClassifier_100.pkl')
    # feature_output = os.path.join(base, 'test.tsv')
    # with open(fastaPath, 'w') as files:
    #     files.write('>test\n')
    #     files.write(seq)

    # GetFeature.getFeature(fastaPath, feature_output, feature)
    # GetFeature.getFeature(fastaPath, feature_output, feature)

    utils = Utils.Utils('Test')

    # featureList, uselessY = utils.readFeature(
    #     feature_output, 0)

    featureList = [get_ifeature(seq, feature)]
    # print(featureList)
    X = np.array(featureList)

    result = utils.predict(modelPath, X)

    return result[0]
