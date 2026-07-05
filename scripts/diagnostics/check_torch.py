import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
print('Loading torch...')
try:
    import torch
    print('Torch loaded successfully!', torch.__version__)
except Exception as e:
    import traceback
    traceback.print_exc()
