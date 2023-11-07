import sys
import os
sys.path.append(os.path.dirname(__file__))
import numpy as np
from iFeature.ifeature_func import get_ifeature
from service import Utils as Utils
from service import GetFeature as GetFeature


def amp_feature(seq):
    base = os.path.dirname(__file__)
    # fastaPath = os.path.join(base, 'test.fasta')
    feature = 'CTDD'
    modelPath = os.path.join(base, 'model/RandomForestClassifier_800.pkl')
    # feature_output = os.path.join(base, 'test.tsv')
    # with open(fastaPath, 'w') as files:
    #     files.write('>test\n')
    #     files.write(seq)

    # GetFeature.getFeature(fastaPath, feature_output, feature)
    # GetFeature.getFeature(fastaPath, feature_output, feature)

    utils = Utils.Utils('Test')

    # featureList, uselessY = utils.readFeature(
    #     feature_output, 0)
    # print(featureList)
    featureList = [get_ifeature(seq, feature)]
    X = np.array(featureList)

    result = utils.predict(modelPath, X)

    return 0  if result[0] == 1 else 1
