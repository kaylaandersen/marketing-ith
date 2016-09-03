import pandas as pd

# input - move to command line eventually
data_source = r'../data/Ibotta_Marketing_Sr_Analyst_Dataset.csv'

# read data
df = pd.read_csv(data_source)

# data cleaning
# for each day column, split off the time spent in seconds
for col in [x for x in df.columns if x.startswith('day')]:
    # session data
    df[col + '_ts'] = df[col].apply(lambda x: int(x.lstrip('session'))
                                              if 'session' in x else 0)
    # remove trailing chars
    df[col] = df[col].apply(lambda x: x.rstrip('1234567890')
                                      if 'session' in x else x)
    # funded/self funded data
    df[col + '_fund'] = df[col].apply(lambda x: x.lstrip('verify')
                                                if 'verify' in x
                                                else x.lstrip('engage'))
    df[col] = df[col].apply(lambda x: x.rstrip('selffunded')
                                      if 'selffunded' in x
                                      else x.rstrip('funded'))

# eda

# cleaning for any ensemble learning method
def clean_ensemble(df):
    # 1. Change day of week to ordinal variable
    # 2. group count of redemptions into one column
    # 3. group count of unlocks into one column
    # 4. group count of sessions into one column
    # 5. group count of gaps into one column
    # 6. group count of funded/self-funded
    # 6. tally session length in minutes-- add estimate for 
    pass
