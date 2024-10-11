import pandas as pd
import os

def table_to_text(raw_df, target, group):
    def helper(row):
        feature = ''
        for col, value in row.items():
            if col != target and col != group:
                # check if the value is na
                if pd.isna(value) == False:                
                    if col in ['class of worker', 'disability', 'employment status of parents', 
                            'citizenship', 'mobility', 'military service', 
                            'hearing difficulty', 'vision difficulty', 'cognitive difficulty',
                            'grandparent living with grandchildren', 
                            'gave birth to child within the past 12 months',
                            'employment status', ]:
                        feature += "The person {}. ".format(value)    
                    else:
                        feature+= "The {} is {}. ".format(col, value)
        return feature[:-1]
    raw_df['feature'] = raw_df.apply(helper, axis = 1)
    return raw_df[['feature', target, group]]


def serialize_table(data_name, raw_df, target, group, root_dir):
    path = root_dir + '/serialize/{}.pkl'.format(data_name)
    # check if serialized data exists
    if os.path.exists(path) == False:
        df = table_to_text(raw_df, target, group)
        df.to_pickle(path)
        return df
    else:
        return pd.read_pickle(path)