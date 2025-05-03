# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 07:02:11 2025

# Clone the repository and checkout the develop branch:
git clone <repository_url>
cd <repository_directory>
git checkout develop

# Create a new branch for your feature:
git checkout -b feature/new-function

# Make your changes and add the new function

# Stage the changes:
git add .

# Commit the changes:
git commit -m "Add new function"

# Push the changes to the remote repository:
git push origin feature/new-function

Create a pull request:
1. Go to your repository on GitHub (or your Git hosting service).
2. Navigate to the "Pull requests" tab.
3. Click on "New pull request".
4. Select the develop branch as the base branch and your feature/new-function branch as the compare branch.
5. Add a title and description for your pull request.
6. Click on "Create pull request".


@author: TEichner
"""

import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import numpy as np
import math
import seaborn as sns
from bokeh.plotting import figure, show, output_file
from bokeh.io import push_notebook
from bokeh.models import ColumnDataSource
import logging


# Configure logging
logging.basicConfig(filename='error_log.txt', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

'''

This is  a new function

'''

# User-defined exception for file not found
class FileNotFoundError(Exception):
    """
    Custom exception class for file not found errors.

    Attributes
    ----------
    file_name : str
        The name of the file that was not found.
    """
    def __init__(self, file_name):
        """
        Initialize the FileNotFoundError class with the name of the missing file.

        Parameters
        ----------
        file_name : str
            The name of the file that was not found.
        """
        self.file_name = file_name
        super().__init__(f"The file {file_name} was not found.")

    def log_error(self):
        """
        Log the file not found error message to a log file.

        This method logs the error message to a file named 'error_log.txt'
        using the logging module
        """
        logging.error(f"FileNotFoundError: {self.file_name} was not found.")

    def save_error_details(self):
        """
        Save the file not found error details to a separate file.

        This method appends the error message to a file named 'error_details.txt'
        for further analysis and record-keeping.
        """
        with open("error_details.txt", "a") as details_file:
            details_file.write(f"FileNotFoundError: {self.file_name} was not found.\n")

# User-defined exception for database errors
class DatabaseError(Exception):

    """
    Custom exception class for database-related errors.

    Attributes
    ----------
    message : str
        The error message describing the database error.
    """
    def __init__(self, message):
        """
        Initialize the DatabaseError class with an error message.

        Parameters
        ----------
        message : str
            The error message describing the database error.
        """
        self.message = message
        super().__init__(f"Database error: {message}")

    def log_error(self):
        """
        Log the database error message to a log file.

        This method logs the error message to a file named 'error_log.txt'
        using the logging module.
        """
        logging.error(f"DatabaseError: {self.message}")

    def save_error_details(self):
        """
        Save the database error details to a separate file.

        This method appends the error message to a file named 'error_details.txt'
        for further analysis and record-keeping.
        """
        with open("error_details.txt", "a") as details_file:
            details_file.write(f"DatabaseError: {self.message}\n")

class DataHandler:
    def __init__(self, db_name='sqldb.db'):
        """
        Initialize the DataHandler class.

        Parameters
        ----------
        db_name : str, optional
            The name of the SQLite database file (default is 'sqldb.db').
        """
        self.db_name = db_name

    def read_csv(self, file_name):
        """
        Read a CSV file into a pandas DataFrame.

        Parameters
        ----------
        file_name : str
            The name of the CSV file to be read.

        Returns
        -------
        pandas.DataFrame
            The DataFrame containing the data from the CSV file.
        
        Raises
        ------
        FileNotFoundError
            If the specified CSV file is not found. Logs the error and saves error details to a file.
        Exception
            If any other error occurs while reading the CSV file. Logs the error and raises a generic exception.
        """
        try:
            return pd.read_csv(file_name)
        except FileNotFoundError:
            error = FileNotFoundError(file_name)
            error.log_error()
            error.save_error_details()
            raise error
        except Exception as e:
            logging.error(f"An error occurred while reading the CSV file: {e}")
            raise Exception(f"An error occurred while reading the CSV file: {e}")

    def sql_write(self, df, db_table):
        """
        Write a pandas DataFrame to an SQLite database table.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to be written to the SQLite database.
        db_table : str
            The name of the database table where the data will be written.

        Raises
        ------
        DatabaseError
            If an error occurs while writing to the database. Logs the error and saves error details to a file.
        """
        db_connection = sqlite3.connect(self.db_name)
        try:
            df.to_sql(db_table, db_connection, if_exists='replace', index=False)
        except Exception as e:
            error = DatabaseError(f"An error occurred while writing to the database: {e}")
            error.log_error()
            error.save_error_details()
            raise error
        finally:
            db_connection.close()

    def sql_read(self, db_table):
        """
        Read data from an SQLite database table into a pandas DataFrame.

        Parameters
        ----------
        db_table : str
            The name of the database table to be read.

        Returns
        -------
        pandas.DataFrame
            The DataFrame containing the data from the database table.

        Raises
        ------
        DatabaseError
            If an error occurs while reading from the database. Logs the error and saves error details to a file.
        """
        db_connection = sqlite3.connect(self.db_name)
        query = f"SELECT * FROM {db_table}"

        try:
            df_sql = pd.read_sql(query, db_connection)
            return df_sql
        except Exception as e:
            error = DatabaseError(f"An error occurred while reading from the database: {e}")
            error.log_error()
            error.save_error_details()
            raise error
        finally:
            db_connection.close()

    def normalization(self, df):
        """
        Normalize the columns of a pandas DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the data to be normalized.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with normalized data.

        Raises
        ------
        ValueError
            If an error occurs during normalization. Logs the error and raises a ValueError with the error message.

        Normalization is performed as follows:
        - If a column contains both positive and negative values, it is normalized to the range [-1, 1].
        - If a column contains only positive values, it is normalized to the range [0, 1].
        - If a column contains only negative values, it is normalized to the range [-1, 0].
        - If a column has constant values, it is set to 0 to avoid division by zero.
        
        Normalization formula:
        - For columns with both positive and negative values:
          normalized_value = 2 * ((value - min_value) / (max_value - min_value)) - 1
        - For columns with only positive values:
          normalized_value = (value - min_value) / (max_value - min_value)
        - For columns with only negative values:
          normalized_value = (value - max_value) / (max_value - min_value)
        - For columns with constant values:
          normalized_value = 0
        """
        try:
            df_norm = df.copy()
            for col in df_norm.columns[1:]:
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()

                if (min_val < 0 and max_val > 0):
                    df_norm[col] = 2 * ((df_norm[col] - min_val) / (max_val - min_val)) - 1
                elif min_val >= 0:
                    df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
                elif max_val <= 0:
                    df_norm[col] = (df_norm[col] - max_val) / (max_val - min_val)
                else:
                    df_norm[col] = 0
            return df_norm

        except Exception as e:
            logging.error(f"An error occurred during normalization: {e}")
            raise ValueError(f"An error occurred during normalization: {e}")

class Plotter:
    def __init__(self):
        """
        Initialize the Plotter class.
        """
        pass

    def individual_plots(self, df):
        """
        Plot individual columns of the DataFrame against the first column using 
        Matplotlib and Bokeh.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the data to be plotted.

        Returns
        -------
        None
        """
        x = df.iloc[:, 0]
        
        # Matplotlib plots
        for column in df.columns[1:]:
            plt.figure(figsize=(10, 6))
            plt.scatter(x, df[column], label=column)
            plt.xlabel('x-values')
            plt.ylabel('y-values')
            plt.legend()
            #plt.show()
            plt.close()

        # Bokeh plots
        for column in df.columns[1:]:
            p = figure(
#                title=f"{column} vs {df.columns}",
                x_axis_label='x-values', 
                y_axis_label='y-values')
            p.scatter(x, df[column], legend_label=column)
            output_file(f"{column}.html")
#            show(p, notebook_handle=True)
#            push_notebook()

    def overlay_plots(self, df1, df1_column, df2, df2_column):
        """
        Overlay plots of selected columns from two DataFrames.

        Parameters
        ----------
        df1 : pandas.DataFrame
            The first DataFrame containing the data to be plotted.
        df1_column : str
            The column name from the first DataFrame to be plotted.
        df2 : pandas.DataFrame
            The second DataFrame containing the data to be plotted.
        df2_column : str
            The column name from the second DataFrame to be plotted.

        Returns
        -------
        None
        """
        df_merge = pd.concat([df1[df1_column], df2[df2_column]], axis=1)
        
        x = df1.iloc[:, 0]
        plt.figure(figsize=(10, 6))
        for column in df_merge.columns:
            plt.scatter(x, df_merge[column], label=column)
            plt.xlabel('x-values')
            plt.ylabel("y-values")
            plt.legend()
        plt.show()
        #plt.close()

class Analyzer:
    def __init__(self, data_handler):
        """
        Initialize the Analyzer class.

        Parameters
        ----------
        data_handler : DataHandler
            An instance of the DataHandler class to interact with data.
        """
        self.data_handler = data_handler

    def compute_deviation(self, train_norm_sql, ideal_norm_sql):
        """
        Compute the deviation between normalized train and ideal datasets.

        Parameters
        ----------
        train_norm_sql : pandas.DataFrame
            The normalized train dataset.
        ideal_norm_sql : pandas.DataFrame
            The normalized ideal dataset.

        Returns
        -------
        tuple
            A tuple containing two lists:
            - ideal_4: List of column names from the ideal dataset that have the lowest sum of squared differences with the train dataset.
            - ideal_4_largest_deviation: List of largest deviations for the corresponding columns in ideal_4.
        """
        ideal_4, ideal_4_largest_deviation = [], []
        for train_column in train_norm_sql.columns[1:]:
            masterlist = []
            for ideal_column in ideal_norm_sql.columns[1:]:
                least_squares = np.sum((train_norm_sql[train_column] - ideal_norm_sql[ideal_column]) ** 2)
                largest_deviation = np.max(np.abs(train_norm_sql[train_column] - ideal_norm_sql[ideal_column]))
                masterlist.append((train_column, ideal_column, least_squares, largest_deviation))
            min_tuple = min(masterlist, key=lambda x: x[2])
            print(min_tuple)
            ideal_4.append(min_tuple[1])  # Append only the column name
            ideal_4_largest_deviation.append(min_tuple[3])
        
        return ideal_4, ideal_4_largest_deviation

    def map_test_to_ideal(self, test_norm_sql, ideal_norm_sql, ideal_4):
        """
        Map the test dataset to the corresponding columns in the ideal dataset.

        Parameters
        ----------
        test_norm_sql : pandas.DataFrame
            The normalized test dataset.
        ideal_norm_sql : pandas.DataFrame
            The normalized ideal dataset.
        ideal_4 : list
            List of column names from the ideal dataset that have the lowest sum of squared differences with the train dataset.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the merged test and corresponding ideal columns.
        """
        ideal_x_values = ideal_norm_sql.iloc[:, 0]
        ideal_y_values_in_tuples = list(zip(*(ideal_norm_sql[col] for col in ideal_4)))
        test_norm_sql_clean = test_norm_sql.groupby('x', as_index=False)['y'].mean()
        list_of_ideal_tuples = []
        for test_value in test_norm_sql_clean['x']:
            for ideal_value in range(len(ideal_x_values)):
                if round(test_value, 1) == round(ideal_x_values[ideal_value], 1):
                    list_of_ideal_tuples.append(ideal_y_values_in_tuples[ideal_value])
        ideal_norm_sql_clean = pd.DataFrame(list_of_ideal_tuples, columns=ideal_4)
        test_ideal_merge = pd.concat([test_norm_sql_clean, ideal_norm_sql_clean], axis=1)
        
        return test_ideal_merge

    def compute_test_deviation(self, test_ideal_merge, ideal_4, ideal_4_largest_deviation):
        """
        Compute the deviation between the test dataset and corresponding columns in the ideal dataset.

        Parameters
        ----------
        test_ideal_merge : pandas.DataFrame
            The merged test and corresponding ideal columns.
        ideal_4 : list
            List of column names from the ideal dataset that have the lowest sum of squared differences with the train dataset.
        ideal_4_largest_deviation : list
            List of largest deviations for the corresponding columns in ideal_4.

        Returns
        -------
        None
        """
        for col in ideal_4:
            test_ideal_merge[f'dev_{col}'] = (test_ideal_merge['y'] - test_ideal_merge[col]).abs()
        count_lst = []
        for fct, threshold in zip(ideal_4, ideal_4_largest_deviation):
            count = (test_ideal_merge['dev_' + fct] < math.sqrt(2) * threshold).sum()
            print(count)
            count_lst.append(count)

        return count_lst

def main():
    """
    Main function to execute the data processing and analysis workflow.

    This function performs the following steps:
    1. Initializes instances of DataHandler, Plotter, and Analyzer classes.
    2. Reads CSV files (train, ideal, test) and writes them to an SQLite database.
    3. Normalizes the data and writes the normalized data to the SQLite database.
    4. Generates individual plots for the normalized train, ideal, and test datasets.
    5. Computes the deviation between the normalized train and ideal datasets.
    6. Generates overlay plots for the matched train and ideal datasets.
    7. Maps the test dataset to the corresponding columns in the ideal dataset.
    8. Writes the merged test and ideal dataset to the SQLite database.
    9. Generates a scatter plot for the merged test and ideal dataset.
    10. Computes the deviation between the test dataset and corresponding columns in the ideal dataset.

    Returns
    -------
    None
    """
    # Initialize instances of DataHandler, Plotter, and Analyzer classes
    data_handler = DataHandler()
    plotter = Plotter()
    analyzer = Analyzer(data_handler)

    # Read CSV files (train, ideal, test) and write them to an SQLite database
    data_handler.sql_write(data_handler.read_csv('train.csv'), 'train')
    data_handler.sql_write(data_handler.read_csv('ideal.csv'), 'ideal')
    data_handler.sql_write(data_handler.read_csv('test.csv'), 'test')

    # Normalize the data and write the normalized data to the SQLite database
    data_handler.sql_write(data_handler.normalization(data_handler.sql_read('train')), 'train_norm')
    data_handler.sql_write(data_handler.normalization(data_handler.sql_read('ideal')), 'ideal_norm')
    data_handler.sql_write(data_handler.normalization(data_handler.sql_read('test')), 'test_norm')

    # Generate individual plots for the original and normalized train, ideal, and test datasets
    plotter.individual_plots(data_handler.sql_read('train'))
    plotter.individual_plots(data_handler.sql_read('train_norm'))
    plotter.individual_plots(data_handler.sql_read('ideal'))
    plotter.individual_plots(data_handler.sql_read('ideal_norm'))
    plotter.individual_plots(data_handler.sql_read('test'))
    plotter.individual_plots(data_handler.sql_read('test_norm'))

    # Compute the deviation between the normalized train and ideal datasets
    train_norm_sql = data_handler.sql_read('train_norm')
    ideal_norm_sql = data_handler.sql_read('ideal_norm')
    ideal_4, ideal_4_largest_deviation = analyzer.compute_deviation(train_norm_sql, ideal_norm_sql)

    # Generate overlay plots for the matched train and ideal datasets
    for count in range(len(ideal_4)):
        plotter.overlay_plots(train_norm_sql, train_norm_sql.columns[count + 1], ideal_norm_sql, ideal_4[count])

    # Map the test dataset to the corresponding columns in the ideal dataset
    test_norm_sql = data_handler.sql_read('test_norm')
    test_ideal_merge = analyzer.map_test_to_ideal(test_norm_sql, ideal_norm_sql, ideal_4)
    
    # Write the merged test and ideal dataset to the SQLite database
    data_handler.sql_write(test_ideal_merge, 'test_ideal_merge')

    # Generate a scatter plot for the merged test and ideal dataset
    colors = sns.color_palette("inferno", n_colors=len(test_ideal_merge.columns) - 1)
    plt.figure(figsize=(10, 6))
    for i, column in enumerate(test_ideal_merge.columns[1:]):
        plt.scatter(test_ideal_merge.iloc[:, 0], test_ideal_merge[column], label=column, color=colors[i])
    plt.xlabel('x-values')
    plt.ylabel('y-values')
    plt.legend()
    plt.show()

    # Compute the deviation between the test dataset and corresponding columns in the ideal dataset
    analyzer.compute_test_deviation(test_ideal_merge, ideal_4, ideal_4_largest_deviation)

if __name__ == '__main__':
    main()