''' nothing to look here '''
# scroll on !!!
import os

# MODULE_PATH = os.path.dirname(__file__)
# PROJECT_ROOT = os.path.dirname(MODULE_PATH)
PROJECT_ROOT = os.getcwd()
BASE_PATH = os.path.join(PROJECT_ROOT, 'data')
CLASS_MAP = {
    0: 'G11', 1: 'G15', 2: 'G17', 3: 'G19', 4: 'G32', 5: 'G34', 6: 'G48', 7: 'G49'
}

try:
    os.makedirs(os.path.join(PROJECT_ROOT, 'weights'))
except OSError:
    pass
