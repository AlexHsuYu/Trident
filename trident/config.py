import os
from . import BASE_PATH

class Config():
    ''' Rand Forest Stuff '''
    TRAIN_PATH = os.path.join(BASE_PATH, 'stage1_train')
    TEST_PATH = os.path.join(BASE_PATH, 'stage1_test')
    FEATURE_SIZE = 20
