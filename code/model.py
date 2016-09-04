import pandas as pd

# input - move to command line eventually
data_source = r'../data/Ibotta_Marketing_Sr_Analyst_Dataset.csv'

# read data
df = pd.read_csv(data_source, index_col='customer_id')

def clean_info(df):
    df = df.copy()
    # convert day of week to ordinal
    dow = {'monday': 0, 'tuesday': 1, 'wednesday':2, 'thursday': 3, 'friday': 4,
    'saturday': 5, 'sunday': 6}
    df['start_dow'] = df['start_day'].apply(lambda x: dow[x])
    # for each day column, split off the time spent in seconds
    for col in [x for x in df.columns if x.startswith('day')]:
        # session data
        df[col + '_ts'] = df[col].apply(lambda x: int(x.lstrip('session'))
                                                  if 'session' in x else 0)
        # remove trailing chars
        df[col] = df[col].apply(lambda x: x.rstrip('1234567890')
                                          if 'session' in x else x)
        # funded/self funded data
        df[col + '_fund'] = df[df[col].str.startswith('verify') |
                               df[col].str.startswith('engage')][col].apply(
                               lambda x: x.replace('verify', '')
                                         if 'verify' in x
                                         else x.replace('engage', ''))
        df[col] = df[col].apply(lambda x: x.replace('selffunded', '')
                                          if 'selffunded' in x
                                          else x.replace('funded', ''))
    return df

def stack_days(df, columns, column_name):
    dfc = df.loc[:, columns]
    dfc.rename(columns={c : int(''.join([char for char in c if char.isdigit()]))
                        for c in dfc.columns if 'day' in c}, inplace=True)
    stack = dfc.stack(dropna=False)
    stack.index.set_names('day', level=-1, inplace=True)
    stack_frame = stack.to_frame(name=column_name)
    return stack_frame

def dow(df):
    dfc = df.loc[:, ['start_dow']]
    for day in range(1, 15):
        dfc[day] = dfc['start_dow'].apply(lambda x: (x + day - 1) % 7)
    dfc.drop('start_dow', axis=1, inplace=True)
    stack = dfc.stack(dropna=False)
    stack.index.set_names('day', level=-1, inplace=True)
    stack_frame = stack.to_frame(name='dow')
    return stack_frame

# split out the time and funding data from the activity into new columns
clean_df = clean_info(df)

# create a dataframe with the activity type by customer and day
act_cols = [c for c in clean_df.columns if 'day' in c and '_' not in c]
act_df = stack_days(clean_df, act_cols, 'activity')

# create a dataframe with time spent in app by customer and day
ts_cols = [c for c in clean_df.columns if '_ts' in c]
ts_df = stack_days(clean_df, ts_cols, 'time_spent')

# create a dataframe with the funding type by customer and day
fund_cols = [c for c in clean_df.columns if '_fund' in c]
fund_df = stack_days(clean_df, fund_cols, 'funded')

dow_df = dow(clean_df)

join_df = act_df.join(ts_df, how='inner').join(fund_df, how='inner')
join_df_dow = join_df.join(dow_df, how='inner')
join_df_dow.set_index('dow', append=True, inplace=True)

# feature engineering!

sum_act = join_df['activity'].groupby(level=0).value_counts().unstack().fillna(0)
sum_day = join_df['activity'].groupby(level=1).value_counts().unstack().fillna(0)
sum_dow = join_df_dow['activity'].groupby(level=2).value_counts().unstack().fillna(0)
