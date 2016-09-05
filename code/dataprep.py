import numpy as np
import pandas as pd
import sys

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

def main(data_csv):

    # read data
    df = pd.read_csv(data_csv, index_col='customer_id')

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

    # feature engineering!
    # adds a label for each activity type and the number of days of each
    sum_act = act_df['activity'].groupby(level=0).value_counts().unstack().fillna(0)
    # column for total time spent in app (in seconds)
    sum_ts = ts_df['time_spent'].astype(float).groupby(level=0).sum().fillna(0)
    # sum of count by each funding type-- this might need work as a feature
    sum_fund = fund_df['funded'].groupby(level=0).value_counts().unstack()

    # get days until app first opened- any activity
    days = act_df.reset_index(level=1)
    act_days_series = days.query("activity != 'gap'")['day'].groupby(level=0).apply(
                                                              lambda x: np.array(x))
    # Use the days the user was active to add first activity day and longest streak
    act_days_df = act_days_series.to_frame('days_active')
    act_days_df['first_activity'] = act_days_df['days_active'].apply(lambda x: x[0])
    act_days_df['streak_activity'] = act_days_df['days_active'].apply(lambda x:
                 max([len(i) for i in np.split(x, np.where(np.diff(x) != 1)[0]+1)]))
    # fill in na's-- these will be ones where they were gaps all 14 days
    act_first = clean_df.loc[:, []].join(act_days_df).fillna({'first_activity': 15,
                                                              'streak_activity': 0},
                                                              axis=0)
    act_first.drop('days_active', axis=1, inplace=True)

    # Join final
    final_df = clean_df[['start_dow']].join(sum_act).join(sum_ts).join(sum_fund).join(act_first).join(clean_df[['future_redemptions']])
    # filling in the nas with 0 for now- I would prob do something better though***s
    final_df.fillna({'funded': 0, 'selffunded': 0}, inplace=True)
    return final_df

    # # might do somthing with the days later
    # add dow for each day
    #dow_df = dow(clean_df)
    # sum_day = join_df['activity'].groupby(level=1).value_counts().unstack().fillna(0)
    # sum_dow = join_df_dow['activity'].groupby(level=2).value_counts().unstack().fillna(0)

if __name__ == '__main__':

    # input - move to command line eventually
    #data_csv = r'../data/Ibotta_Marketing_Sr_Analyst_Dataset.csv'

    model_df = main(sys.argv[1])
