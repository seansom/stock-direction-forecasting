# Simple script used to delete databases. Used when changing the specific direction forecasting files used for testing

import os, shutil

# delete params and model_info databases
os.chdir('data')
for filename in os.listdir():
    if filename.startswith('model_info_database') or filename.startswith('stock_database'):
        os.remove(filename)
os.chdir('..')

# delete saved models
if os.path.exists('models'):
    shutil.rmtree('models')
    os.makedirs('models')