import os
import mlflow.data.numpy_dataset
import mlflow.models.evaluation
from sklearn import metrics,ensemble,linear_model,svm,model_selection,dummy,tree
import pandas as pd 
import numpy as np 
import xgboost 
import mlflow 
from params import xgb_grid,rf_grid,adaboost_grid,stacking_grid,dt_grid
from utils import evaluate,acc_metric,f1_metric,precision_metric,recall_metric,roc_metric,print_dct
import warnings
warnings.filterwarnings('ignore')

training_data=pd.read_csv('data/train_data_processed.csv')
validation_data=pd.read_csv('data/val_data_processed.csv')

ind_arr=[col for col in training_data.columns if col!='class']
X_train,y_train=training_data[ind_arr],training_data['class']
X_val,y_val=validation_data[ind_arr],validation_data['class']

mlflow.set_tracking_uri('http://127.0.0.1:5000')

experiments_list=['Decision Tree Mushroom Training','Random Forest Mushroom Training','XGBoost Mushroom Trianing','Stacking Classiifer Mushroom Training','Adaboost Mushroom Training']

exp_list=list(zip(experiments_list,[dt_grid,rf_grid,xgb_grid,stacking_grid,adaboost_grid]))

print('Base Classifier Training...')
base_clf=dummy.DummyClassifier()
base_clf.fit(X_train,y_train)
print_dct(evaluate(base_clf.predict(X_val),y_val))
print('Classifier Trianing...')
for exp_name,parameters in exp_list:
    print(f'{exp_name} training begun')
    exp=mlflow.set_experiment(experiment_name=exp_name)
    exp_id=exp.experiment_id
    for params in model_selection.ParameterGrid(parameters):
        mlflow.end_run()
        with mlflow.start_run(experiment_id=exp_id):
            mlflow.sklearn.log_model(base_clf,'base-classifier')
            base_uri=mlflow.get_artifact_uri('base-classifier')
            if exp_name[0].lower()=='x':
                clf=xgboost.XGBClassifier(**params)
            elif exp_name[0].lower()=='r':
                clf=ensemble.RandomForestClassifier(**params)
            elif exp_name[0].lower()=='a':
                clf=ensemble.AdaBoostClassifier(**params)
            elif exp_name[0].lower()=='s':
                clf=ensemble.StackingClassifier(estimators=[ensemble.RandomForestClassifier(),svm.SVC()],
                                                final_estimator=linear_model.LogisticRegression(),**params)
            else:
                clf=tree.DecisionTreeClassifier(**params)

            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_val)
            evaluations=evaluate(y_pred,y_val)
            print_dct(evaluate(clf.predict(X_val),y_val))
            mlflow.log_params(params)
            mlflow.log_metrics(evaluations)
            mlflow.sklearn.log_model(clf,exp_name.split(' ')[0],
                                    input_example=X_train,
                                    code_paths=['train.py'])
            clf_uri=mlflow.get_artifact_uri(exp_name.split(' ')[0])
            print('Evaluation phase..')
            thresholds = {"f1_score": mlflow.models.evaluation.MetricThreshold(threshold=0.5,
                                                                               min_absolute_change=0.01,
                                                                               min_relative_change=0.01,
                                                                               greater_is_better=True),
                          "accuracy_score": mlflow.models.evaluation.MetricThreshold(threshold=0.6,
                                                                                     min_absolute_change=0.01,
                                                                                     min_relative_change=0.01,
                                                                                     greater_is_better=True)
                          }

            try:
                result = mlflow.evaluate(
                    model=clf_uri,
                    data=validation_data,
                    targets="class",
                    model_type="classifier",
                    custom_metrics=[
                        mlflow.models.make_metric(eval_fn=acc_metric, greater_is_better=True, name='accuracy_score'),
                        mlflow.models.make_metric(eval_fn=precision_metric, greater_is_better=True, name='precision_score'),
                        mlflow.models.make_metric(eval_fn=recall_metric, greater_is_better=True, name='recall_score'),
                        mlflow.models.make_metric(eval_fn=f1_metric, greater_is_better=True, name='f1_score'),
                    ],
                    validation_thresholds=thresholds,
                    baseline_model=base_uri,
                )
            except ValueError as e:
                if "The beeswarm plot does not support plotting explanations with instances that have more than one dimension" in str(e):
                    print("Skipping beeswarm plot due to multidimensional instances issue.")
                else:
                    raise
            print('run completed..')
            
            

