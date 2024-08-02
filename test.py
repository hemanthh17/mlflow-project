import os 
import pandas as pd 
import numpy as np
import sklearn
import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
test_data=pd.read_csv('data/test_data_processed.csv')
inference_model=mlflow.pyfunc.load_model("runs:/2f048c1f71094eb3abf07e84f31f4507/Decision")

y_test_pred=inference_model.predict(test_data)
y_test_corr=np.array(list(map(lambda x:'e' if x==0 else 'p',y_test_pred)))
sub_data=pd.read_csv('data/sample_submission.csv')
sub_data['class']=y_test_corr
sub_data.to_csv('data/submission.csv',index=False)

