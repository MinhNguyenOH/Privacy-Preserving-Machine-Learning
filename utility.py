import argparse
import os


def parseargs(arglist):
    ''' Argument parser
        Parameters:
            arglist: the command line list of args
        Returns:
            the result of parsing with the system parser
    '''
    parser = argparse.ArgumentParser()

    for onearg in arglist:
        parser.add_argument(onearg[0], onearg[1], help=onearg[2])
    args = parser.parse_args()

    return args


def check_file_format(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.csv':
        return 'csv'
    elif file_extension.lower() in ('.xls', '.xlsx'):
        return 'excel'
    elif file_extension.lower() == '.json':
        return 'json'
    return 'Unknown'
