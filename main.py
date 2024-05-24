'''
Import DecisionTree module, pandas dataframe, and argument parser
'''
import sys
import pandas as pd
from decision_tree import DecisionTree
import utility


def main():
    '''
    Sample main function
    '''

    filename = 'mushroom_data_clean.csv'
    file_format = utility.check_file_format(filename)
    if file_format == 'csv':
        df = pd.read_csv(filename)
    elif file_format == 'excel':
        df = pd.read_excel(filename)
    elif file_format == 'json':
        df = pd.read_json(filename)
    else:
        print('Not supported file format')
        sys.exit(1)

    # Configure parameters
    # weather data
    # attributes = set(['Outlook', 'Temperature', 'Humidity', 'Wind'])
    # label = 'PlayTennis'

    # breast cancer data
    # attributes = set(['age', 'tumor-size', 'menopause', 'node-caps'])
    # label = 'irradiat'

    # mushroom data
    attributes = set(['cap-shape', 'cap-surface', 'cap-color', 'veil-type', 'veil-color'])
    label = 'poisonous'

    # Changing `epsilon` and `depth` may change accuracy
    epsilon = 5
    depth = 1

    decision_tree = DecisionTree()
    decision_tree.build_tree(df, attributes, label, epsilon, depth)

    # Evaluate the model accuracy
    match_count = 0
    for i in range(len(df)):
        record = df.iloc[i]
        result = decision_tree.predict(record)
        # print(result, record[label])
        if result == record[label]:
            match_count += 1

    print('Accuracy:', match_count / len(df))


if __name__ == "__main__":
    main()
