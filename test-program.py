# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 07:16:13 2025

@author: TEichner
"""

import unittest
import pandas as pd
import numpy as np
from io import StringIO
from oop_least_squares import DataHandler, Plotter, Analyzer  

class TestDataHandler(unittest.TestCase):
    def setUp(self):
        self.db_name = 'test_db.sqlite' # Use a file-based SQLite database for testing
        self.data_handler = DataHandler(db_name=self.db_name)
        self.csv_data = """x,y1,y2,y3,y4
        1,-5,15,-25,3
        2,0,0,0,0
        3,5,15,-25,-1"""
        self.df = pd.read_csv(StringIO(self.csv_data))

    def test_read_csv(self):
        df = self.data_handler.read_csv(StringIO(self.csv_data))
        pd.testing.assert_frame_equal(df, self.df)

    def test_sql_write_and_read(self):
        self.data_handler.sql_write(self.df, 'test_table')
        df_sql = self.data_handler.sql_read('test_table')
        pd.testing.assert_frame_equal(df_sql, self.df)

    def test_normalization(self):
        df_norm = self.data_handler.normalization(self.df)
        expected_data = {
            'x': [1, 2, 3],
            'y1': [-1.0, 0.0, 1.0],  # test in case of positve and negative values (even num ratio between pos and neg)
            'y2': [1.0, 0.0, 1.0],  # test in case of only positive values
            'y3': [-1.0, 0.0, -1.0],  # test in case of only negative values
            'y4': [1.0, -0.5, -1.0]  # test that numerical ratio between pos and neg is conserved
        }
        expected_df = pd.DataFrame(expected_data)
        pd.testing.assert_frame_equal(df_norm, expected_df)

class TestPlotter(unittest.TestCase):
    def setUp(self):
        self.plotter = Plotter()
        self.csv_data = """x,y1,y2
        1,-5,15
        2,0,0
        3,5,15"""
        self.df = pd.read_csv(StringIO(self.csv_data))

    def test_individual_plots(self):
        # This test will check if the method runs without errors
        self.plotter.individual_plots(self.df)

    def test_overlay_plots(self):
        # This test will check if the method runs without errors
        self.plotter.overlay_plots(self.df, 'y1', self.df, 'y2')

class TestAnalyzer(unittest.TestCase):
    def setUp(self):
        self.data_handler = DataHandler(db_name=':memory:')
        self.analyzer = Analyzer(self.data_handler)
        self.train_data = """x,y1,y2,y3,y4
        1,-5,15,-25,3
        2,0,0,0,0
        3,5,15,-25,-1"""
        self.ideal_data = """x,y1,y2,y3,y4
        1,-5.5,15.5,-25.5,3.5
        2,0,0,0,0
        3,4.5,14.5,-24.5,-0.5"""
        self.test_data = """x,y
        1,2
        2,3
        3,4"""
        self.train_df = pd.read_csv(StringIO(self.train_data))
        self.ideal_df = pd.read_csv(StringIO(self.ideal_data))
        self.test_df = pd.read_csv(StringIO(self.test_data))

    def test_compute_deviation(self):
        train_norm = self.data_handler.normalization(self.train_df)
        ideal_norm = self.data_handler.normalization(self.ideal_df)
        ideal_4, ideal_4_largest_deviation = self.analyzer.compute_deviation(train_norm, ideal_norm)
        self.assertEqual(ideal_4, ['y1', 'y2', 'y3', 'y4'])
        self.assertEqual(len(ideal_4_largest_deviation), 4)

    def test_map_test_to_ideal(self):
        test_norm = self.data_handler.normalization(self.test_df)
        ideal_norm = self.data_handler.normalization(self.ideal_df)
        ideal_4 = ['y1', 'y4']
        test_ideal_merge = self.analyzer.map_test_to_ideal(test_norm, ideal_norm, ideal_4)
        self.assertIn('y1', test_ideal_merge.columns)
        self.assertIn('y4', test_ideal_merge.columns)

    def test_compute_test_deviation(self):
        test_norm = self.data_handler.normalization(self.test_df)
        ideal_norm = self.data_handler.normalization(self.ideal_df)
        ideal_4 = ['y1', 'y2', 'y3', 'y4']
        test_ideal_merge = self.analyzer.map_test_to_ideal(test_norm, ideal_norm, ideal_4)
        count_lst = self.analyzer.compute_test_deviation(test_ideal_merge, ideal_4, [1.0, 0.5, 0.5, 0.1])
        self.assertIn('dev_y1', test_ideal_merge.columns)
        self.assertIn('dev_y2', test_ideal_merge.columns)
        self.assertEqual(len(count_lst), 4)

if __name__ == '__main__':
    unittest.main()
