import os

def getFeature(input, output, type):
    path_to_ifeature = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    'iFeature/iFeature.py')
    feature = os.system('python3 '+path_to_ifeature+ ' --file ' + input + ' --type ' + type + ' --out ' + output)
    return feature