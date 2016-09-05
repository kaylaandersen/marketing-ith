from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn import grid_search, metrics
import dataprep
from grid_search_params import grid_search_params

# class skeleton
class Model(object):
    """
    This class holds the skelton for common model implementations.
    Inlcudes ability to grid search, fitting and pickling procedures.
    Suggested order of operations:
    1. Pass sklearn model class (not instanstiated)
    2. Run grid search on X and y training data. This will update the model
       hyperparameters used for the next functions to the best estimator from
       the gridsearch.
    3. Predict model. Returns y_pred.
    4. Score Mmdel.
    5. (Optional) For a model you want to reuse, dump into a pickle file.
    """

    def __init__(self, model_class, **kwargs):
        """Instantiates an sklearn model, with optional parameters.
        Args:
        model_class (class): A sklearn model class, not instanstiated.
        **kwargs: Parameters for the sklearmn model class, if desired.
        """
        self.model = model_class(**kwargs) # instatiate model with params

    def run_grid_search(self, grid_search_dict, X_train, y_train):
        """Runs grid search and sets the self.model to the best estimator.
        Args:
        grid_search_dict (dict): A nested dictionary of hyperparameters to test.
        X_train (array): Training dataset features (2d).
        y_train (array): Training dataset labels (1d).
        """
        print 'grid searching...'
        # run grid search
        self.gs = grid_search.GridSearchCV(self.model, grid_search_dict)
        # fit gs to training data
        self.gs.fit(X_train, y_train)
        # set self.model to the best estimator
        self.model = self.gs.best_estimator_

    def predict_model(self, X_test):
        """Predicts labels from test dataset labels.
        Args:
        X_test (array): Test dataset features (2d).
        """
        self.y_pred = self.model.predict(X_test)
        return self.y_pred

    def score(self, y_true, metric):
        """Scores the current prediction compared to the true values.
        Args:
        y_true (array): Test dataset true values (1d).
        metric (function): Sklearn model function defintion to run.
        """
        return metric(y_true, self.y_pred)

    def dump_model(self, outpath):
        """Saves the current model definition into a pickle file.
        Args:
        outpath (string): Path to save the pickle file to.
        """
        with open(outpath, 'w') as f:
            pickle.dump(self.model, f)

# get clean data
model_df = dataprep.main(r'../data/Ibotta_Marketing_Sr_Analyst_Dataset.csv')
# make future user class labels
model_df['future_user'] = model_df['future_redemptions'].apply(lambda x: 1 if x > 0 else 0)
# drop the future_redemptions column (avoid leakage)
model_df.drop('future_redemptions', axis=1, inplace=True)

# pop out the class labels as a numpy array
y = model_df.pop('future_user').values
# convert features to numpy array
x = model_df.values
# make a 70/30 train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=.70)

# Test randome forest classifier
rf = Model(RandomForestClassifier)
rf.run_grid_search(grid_search_params['RandomForestClassifier'], x_train, y_train)
y_pred = rf.predict_model(x_test)
