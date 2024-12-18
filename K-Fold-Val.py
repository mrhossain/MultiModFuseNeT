import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

if __name__ == "__main__":
    # Load the dataset
    data1 = pd.read_csv('./BMMTC6-Final/train.csv')
    data2 = pd.read_csv('./BMMTC6-Final/test.csv')

    data = pd.concat([data1, data2], axis=0)
    data = data.reset_index(drop=True)
    # Shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)

    # Define the KFold with 5 splits
    kf = KFold(n_splits=5, shuffle=True)

    # Train and test split ratio is 70:30
    i = 1
    for train_index, test_index in kf.split(data):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        print("Train + Test data: ", len(train_data) + len(test_data))

        # Calculate the number of samples for 70% of the training data
        # Split the training data into 70% train and 30% test
        train_data, remaining_data = train_data[:10682], train_data[10682:]
        test_data = pd.concat([test_data, remaining_data], axis=0)
        #Save the train and test data with fold number train_fold_{number}.csv
        train_data.to_csv(f'./BMMTC6-Final/5-fold/train_fold_{i}.csv', index=False)
        test_data.to_csv(f'./BMMTC6-Final/5-fold/test_fold_{i}.csv', index=False)
        i += 1