import pandas as pd
from datetime import datetime

post_data_df = pd.read_csv('facebook-fact-check.csv')

# Dropping some columns to clean our data a little bit.
to_drop = ['Page', 'Post Type', 'Debate', 'Post URL']
post_data_df.drop(to_drop, inplace=True, axis=1)


# A function to clean our dates into UNIX timestamps.
def convert_to_unix(date):
    unix = int(datetime.strptime(date, "%Y-%m-%d").timestamp())
    return unix


# Converting our data to numerical values to be able to use it in training a model.
cleaning_dict = {'Category':
                     {'mainstream': 0,
                      'right': (-1),
                      'left': 1},
                 'Rating':
                     {'no factual content': 0,
                      'mostly true': 1,
                      'mostly false': (-2),
                      'mixture of true and false': (-1),
                      }}

post_data_df['Date Published'] = [convert_to_unix(i) for i in post_data_df['Date Published']]
post_data_df['Category'] = [cleaning_dict['Category'][i] for i in post_data_df['Category']]
post_data_df['Rating'] = [cleaning_dict['Rating'][i] for i in post_data_df['Rating']]

# Replacing missing values with 0
post_data_df = post_data_df.fillna(0)

# Overwriting the csv with updated data.
post_data_df.to_csv('facebook-fact-check.csv', index=False)

