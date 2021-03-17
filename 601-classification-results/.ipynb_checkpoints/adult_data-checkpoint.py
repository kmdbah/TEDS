#########################################################
# adult_data.py
#   - Functions for loading and vectorizing Adult Census Data
#
# These functions were moved into this file
#   for reuse and to reduce Notebook complexity.
#########################################################

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline, make_union

################
# CONSTANTS
################

# The CSVs do not have a header row. 
#   These column names are from `adult.names`.
ADULT_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education',
                 'education_num', 'marital_status', 'occupation',
                 'relationship', 'race', 'sex', 'capital_gain',
                 'capital_loss', 'hours_per_week', 'native_country',
                 'income']

TARGET_COLUMN = 'income'

# Manually determined
CATEGORICAL_COLUMNS = ['workclass', 'education', 'marital_status', 'occupation', 
                       'relationship', 'race', 'sex', 'native_country']

NUMERIC_COLUMNS = ['age', 'fnlwgt', 'education_num', 'capital_gain', 
                   'capital_loss', 'hours_per_week']

ALL_FEATURES = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


################
# FUNCTIONS
################

################################
#### Cleaning & Loading Data


# A verbose keyword parameter can be used to control logging/debug output!
def clean_adult_df(df, drop_unknowns=False, verbose=False):
    """
     Returns a cleaned DataFrame.
     | 48842 instances, mix of continuous and discrete    (train=32561, test=16281)
     | 45222 if instances with unknown values are removed (train=30162, test=15060)
    """
    
    # Remove extra `.` from test set target.
    # Train: '>50k', Test: '>50k.'
    df['income'] = df.income.str.strip('.')
    
    # Note: This was tested to exactly remove those rows
    #   indicated by "unknowns" in `adult.names`
    if drop_unknowns:
        df = df[(df.occupation != '?') & 
                (df.native_country != '?') & 
                (df.workclass != '?')].copy()    

    return df


def load_adult_data(csv_filename, skiprows=None, 
                    clean=True, train_features=None,
                    drop_unknowns=True,
                    verbose=True):
    """ Loader for adult data. 
    Written for consistency across train/test loading. """

    # Setting `header=0` supposes the first row as a header, even if it isn't a header!
    #   We used `header=None`, which does not skip the first row.

    # `skipinitialspace` was added because each record starts with a space in the CSV after the comma (", "):
    #   39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
    adult_df = pd.read_csv(csv_filename, 
                           header=None, names=ADULT_COLUMNS,
                           skipinitialspace=True,
                           skiprows=skiprows)
    
    if clean:
        adult_df = clean_adult_df(adult_df, 
                                  drop_unknowns=drop_unknowns,
                                  verbose=verbose)
    
    return adult_df


def load_train_test(train_file, test_file, verbose=False):
    """Load train/test/all."""

    # Train set
    adult_train_df = load_adult_data(train_file, clean=True)
    if verbose: print('Train shape: ', adult_train_df.shape)

    # Test set: first row is invalid
    adult_test_df = load_adult_data(test_file, skiprows=1, clean=True)
    if verbose: print('Test shape: ', adult_test_df.shape)
    
    return adult_train_df, adult_test_df



################################
# Vectorizing Data

def dummify_features(df, verbose=False):
    """ Returns a df with extracted features for EDA, 
        and a set of added column names. """
    dummies_df = pd.get_dummies(df[CATEGORICAL_COLUMNS])
    dummies_df.columns = dummies_df.columns.str.lower()
    
    if verbose:
        print('Dummies shape: ', dummies_df.shape)
    
    df = pd.concat([df, dummies_df], axis=1)
    
    return df, set(dummies_df.columns)


def vectorize(df, encoders=None):
    """ 
    Returns the vectorized feature X df, y, and dict of pre-fit encoders.
    - Set encoders=None on training set. 
    - For test set, pass in the encoders fit earlier.
    """
    
    # Split into numerical and categorical features
    X_num = df[NUMERIC_COLUMNS]
    X_cat = df[CATEGORICAL_COLUMNS]
    
    # Fit encoders (in this case, dummify categorical features)
    if not encoders:
        encoders = {}
        encoders['categorical'] = OneHotEncoder(sparse=False, dtype=np.int64).fit(X_cat)

    # Transform data using pre-fit encoders
    X_cat_num = encoders['categorical'].transform(X_cat)
    X = np.hstack([X_num, X_cat_num])
    
    ##### Transform target
    y = df[TARGET_COLUMN]
    
    
    ##### Create final column names
    categories = encoders['categorical'].categories_
    all_features = list(X_num.columns)
    
    for col_name,col_cats in zip(list(X_cat.columns), categories):
        for uniq_cat in col_cats:
            all_features.append(f'{col_name}_{uniq_cat}')
    
    ##### Return final DataFrame
    X_df = pd.DataFrame(X, columns=all_features, dtype=np.float)
    
    return X_df, y, encoders
