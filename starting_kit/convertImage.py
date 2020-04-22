"""
Created on Sat Mar 27 2020
Last revised: April 18, 2020
@author: Mouloua Ramdane
"""

model_dir = 'sample_code_submission/'                        # Change the model to a better one once you have one!
result_dir = 'sample_result_submission/' 
problem_dir = 'ingestion_program/'  
score_dir = 'scoring_program/'

from sys import path; path.append(model_dir); path.append(problem_dir); path.append(score_dir); 

import seaborn as sns; sns.set()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import pandas as pd

import pickle
import numpy as np   
from os.path import isfile


import warnings

with warnings.catch_warnings():
	import numpy as np
	from data_manager import DataManager

if __name__=="__main__":


	data_dir = 'public_data_raw_gaiasavers'         
	data_name = 'plankton'
	D = DataManager(data_name, data_dir) 

	X_train = D.data['X_train']
	Y_train = D.data['Y_train'].ravel()

	from data_io import read_as_df
	data = read_as_df(data_dir  + '/' + data_name)                

	for i in np.arange(1300	,len(X_train)):
		B = np.reshape(X_train[i], (100, 100))
		plt.imshow(B, cmap="gray")
		print(i)
		plt.savefig('images_raw/'+data['target'][i]+'/'+'img'+str(i)+'.png')