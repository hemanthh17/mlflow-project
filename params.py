xgb_grid={
    'n_estimators': [100, 200, 300],  # Number of boosting rounds
    'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage to prevent overfitting
    'max_depth': [3, 5, 10, 15, 20],  # Maximum depth of each tree
    'min_child_weight': [1, 2, 3],  # Minimum sum of instance weight needed in a child
    'subsample': [0.8, 0.9, 1.0],  # Fraction of samples used for training
    'colsample_bytree': [0.8, 0.9, 1.0],  # Fraction of features used for training
    'gamma': [0, 0.1, 0.2],  # Minimum loss reduction required to make a further partition on a leaf node
    'reg_alpha': [0, 0.1, 1.0],  # L1 regularization term on weights
    'reg_lambda': [0, 0.1, 1.0],  # L2 regularization term on weights
}

rf_grid={
    'criterion':['gini','log-loss'],
    'max_depth':[3, 15, 20],
    'warm_start':[False,True],
    
}

stacking_grid={
    'cv':[None,5,10]
}

adaboost_grid={
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.1, 0.2]


}
dt_grid={
    'criterion':['gini','log-loss'],
    'max_depth':[200,None]

}