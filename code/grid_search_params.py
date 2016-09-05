
grid_search_params = {
'RandomForestClassifier':{'n_estimators':[100, 500], 'criterion':['gini','entropy'],'max_depth':[None,5,20], 'min_samples_split':[2,6], 'max_features':[7], 'bootstrap':[True], 'n_jobs':[-1], 'random_state':[1],'verbose':[True]},
'GradientBoostingClassifier': {'learning_rate':[0.1,0.001,0.5,0.75,1], 'n_estimators':[100,500,1000], 'subsample':[0.1,0.5,1,1.5], 'min_samples_split':[2,4,6], 'max_depth':[None,5,10,20], 'random_state':[1], 'max_features':[None,3,7,10,15]}}
