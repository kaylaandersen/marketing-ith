
grid_search_params = {
'RandomForestClassifier':{'n_estimators':[100, 500],
                          'criterion':['gini','entropy'],
                          'max_depth':[None,5,20],
                          'min_samples_split':[2,6],
                          'max_features':[7],
                          'bootstrap':[True],
                           'n_jobs':[-1],
                           'random_state':[1],
                           'verbose':[True]},
'GradientBoostingClassifier': {'learning_rate':[0.1,0.01,0.5],
                                'n_estimators':[100,500,1000],
                                'min_samples_split': [2, 100, 500],
                                'max_depth':[None,5,10],
                                'random_state':[1]}}
