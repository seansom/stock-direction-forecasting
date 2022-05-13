from tkinter.messagebox import NO
from turtle import update
from tensorflow import keras, compat
from statistics import mean, stdev
import keras_tuner as kt
import numpy as np
import pandas as pd
import os, sys, math, copy, shutil, shelve, warnings, json
from sklearn.preprocessing import PowerTransformer
from data_processing_app import get_dataset, inverse_transform_data, get_dates_five_years, get_trading_dates
from direction_forecasting_app import CustomCallback, make_lstm_hypermodel, get_optimal_hps, make_lstm_model, forecast_lstm_model, get_lstm_model_perf, print_model_performance, experiment
from PyQt5 import QtWidgets as qtw
from PyQt5.QtCore import QThread
from mainwindow import Ui_MainWindow




class PredictionThread(QThread):
    def __init__(self, main_window):
        QThread.__init__(self)
        self.main_window = main_window
        
    def run(self):
        self.main_window.make_prediction()



class GettingThread(QThread):
    def __init__(self, main_window):
        QThread.__init__(self)
        self.main_window = main_window
        
    def run(self):
        self.main_window.get_model()



class TuningThread(QThread):
    def __init__(self, main_window):
        QThread.__init__(self)
        self.main_window = main_window
        
    def run(self):
        self.main_window.tune_model()



class MainWindow(qtw.QMainWindow):
    """The main window for the program.

    Args:
        qtw (QMainWindow): PyQt5 Main Window Widget
    """
    def __init__(self, app, parent=None):
        super().__init__(parent)

        self.app = app

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setFixedSize(self.size())

        self.setWindowTitle("Stock Direction Forecasting")

        self.ui.predict_button.clicked.connect(self.run_prediction_thread)
        self.ui.get_model_button.clicked.connect(self.run_getting_thread)
        self.ui.tune_model_button.clicked.connect(self.run_tuning_thread)


    def disable_ui(self):
        self.ui.stock_ticker_line_edit1.setEnabled(False)
        self.ui.predict_button.setEnabled(False)
        self.ui.stock_ticker_line_edit2.setEnabled(False)
        self.ui.get_model_button.setEnabled(False)
        self.ui.stock_ticker_line_edit3 .setEnabled(False)
        self.ui.tune_model_button.setEnabled(False)


    def enable_ui(self):
        self.ui.stock_ticker_line_edit1.setEnabled(True)
        self.ui.predict_button.setEnabled(True)
        self.ui.stock_ticker_line_edit2.setEnabled(True)
        self.ui.get_model_button.setEnabled(True)
        self.ui.stock_ticker_line_edit3 .setEnabled(True)
        self.ui.tune_model_button.setEnabled(True)


    def get_hps(self, stock_ticker):

        os.chdir('data')

        hps_database = shelve.open('hps_database')
        hps_database_key = f"{stock_ticker}"

        if hps_database_key in hps_database:
            hps = hps_database[hps_database_key]

        else:
            hps = None

        hps_database.close()
        os.chdir('..')

        return hps


    def get_built_models(self, stock_ticker, last_model_tuning_date):

        os.chdir('data')

        model_database = shelve.open('model_database')
        model_database_key = f"{stock_ticker}"

        if model_database_key in model_database and (last_model_tuning_date == 'Never' or model_database[model_database_key]['training_date'] == last_model_tuning_date):
            models = model_database[model_database_key]

        else:
            models = {}

        model_database.close()
        os.chdir('..')

        return models


    def make_prediction(self):

        self.disable_ui()

        try:
            stock_ticker = self.ui.stock_ticker_line_edit1.text()
            stock_ticker = stock_ticker.strip().upper()

            self.ui.stock_ticker_line_edit1.clear()

            if len(stock_ticker) == 0:
                self.ui.status_label1.setText('[Error] Invalid stock ticker')
                self.ui.status_label2.setText('[Error] Invalid stock ticker')
                self.enable_ui()
                return

            self.ui.status_label1.setText('Please check console for possible confirmation')
            self.ui.status_label2.setText('Please check console for possible confirmation')

            repeats = 5
            performances = []
            final_predictions = []

            hps = self.get_hps(stock_ticker)
            time_steps = hps['time_steps'] if hps is not None else 20

            last_model_tuning_date = hps['last_model_tuning_date'] if hps is not None else 'Never'

            models = self.get_built_models(stock_ticker, last_model_tuning_date)

            if not models:

                print("===================================================")
                for i in range(repeats):
                    print(f"Experiment {i + 1} / {repeats}")
                    
                    scaler, col_names, train_x, train_y, test_x, test_y, final_window = get_dataset(stock_ticker, date_range=get_dates_five_years(init_date=last_model_tuning_date), time_steps=time_steps, drop_col=None)
                    
                    # for when user did not continue with News API historical call, will instead input new stock ticker
                    if scaler is None and col_names is None:
                        self.ui.status_label1.setText('Idle - Please input new stock ticker')
                        self.ui.status_label2.setText('Idle - Please input new stock ticker')
                        self.enable_ui()
                        return

                    self.ui.status_label1.setText(f'Forecasting {stock_ticker} stock...')
                    self.ui.status_label2.setText(f'Forecasting {stock_ticker} stock...')
                    
                    model, perf, _, _, final_prediction = experiment(scaler, col_names, train_x, train_y, test_x, test_y, final_window, hps=hps, window=self)
                    performances.append(perf)
                    final_predictions.append(final_prediction)


                    os.chdir('data')
                    model_database = shelve.open('model_database')
                    model_database_key = f"{stock_ticker}"

                    if model_database_key not in model_database:
                        model_database[model_database_key] = {}

                    model_database[model_database_key][f'model {i}'] = model
                    model_database[model_database_key][f'scaler {i}'] = scaler

                    model_database.close()
                    os.chdir('..')

                    print("===================================================")
                    self.ui.models_progress_label.setText(f'{i + 1} / {repeats}')


                mean_da = round(mean([perf['da'] for perf in performances]), 6)
                mean_uda = round(mean([perf['uda'] for perf in performances]), 6)
                mean_dda = round(mean([perf['dda'] for perf in performances]), 6)

                std_da = round(stdev([perf['da'] for perf in performances]), 6)
                std_uda = round(stdev([perf['uda'] for perf in performances]), 6)
                std_dda = round(stdev([perf['dda'] for perf in performances]), 6)

                mean_total_ups = mean([perf['total_ups'] for perf in performances])
                mean_total_downs = mean([perf['total_downs'] for perf in performances])
                
                optimistic_baseline = round(mean_total_ups / (mean_total_ups + mean_total_downs), 6)
                pessimistic_baseline = round(1 - optimistic_baseline, 6)
                
                os.chdir('data')
                model_database = shelve.open('model_database')
                model_database_key = f"{stock_ticker}"

                training_date = get_dates_five_years(init_date=last_model_tuning_date)[1]
                model_database[model_database_key]['training_date'] = training_date

                model_database[model_database_key]['mean_da'] = mean_da
                model_database[model_database_key]['mean_uda'] = mean_uda
                model_database[model_database_key]['mean_da'] = mean_da
                model_database[model_database_key]['std_da'] = std_da
                model_database[model_database_key]['std_uda'] = std_uda
                model_database[model_database_key]['std_dda'] = std_dda
                model_database[model_database_key]['optimistic_baseline'] = optimistic_baseline
                model_database[model_database_key]['pessimistic_baseline'] = pessimistic_baseline

                model_database.close()
                os.chdir('..')


            else:
                print("===================================================")
                for i in range(repeats):
                    print(f"Experiment {i + 1} / {repeats}")
                    
                    scaler, _, _, _, _, _, final_window = get_dataset(stock_ticker, date_range=get_dates_five_years(), time_steps=time_steps, drop_col=None)

                    final_window = scaler.inverse_transform(np.reshape(final_window, final_window.shape[1:]))

                    os.chdir('data')
                    model_database = shelve.open('model_database')
                    model_database_key = f"{stock_ticker}"

                    model = model_database[model_database_key][f'model {i}']
                    scaler = model_database[model_database_key][f'scaler {i}']

                    model_database.close()
                    os.chdir('..')

                    final_window = np.reshape(scaler.transform(final_window), [1, final_window.shape[0], final_window.shape[1]])

                    final_prediction = forecast_lstm_model(model, final_window)
                    final_prediction = inverse_transform_data(final_prediction, scaler, col_names, feature="log_return")
                    final_prediction = [i.tolist() for i in final_prediction]



            self.ui.last_tuning_date_label1.setText(last_model_tuning_date)
            self.ui.mean_da_label.setText(str(mean_da))
            self.ui.mean_uda_label.setText(str(mean_uda))
            self.ui.mean_dda_label.setText(str(mean_dda))
            self.ui.stdev_da_label.setText(str(std_da))
            self.ui.stdev_uda_label.setText(str(std_uda))
            self.ui.stdev_dda_label.setText(str(std_dda))

            self.ui.result_stock_ticker_label.setText(stock_ticker)

            with open('keys/EOD_API_key.txt') as file:
                token = file.readline()
            last_observed_trading_day = get_trading_dates(stock_ticker, get_dates_five_years(), token).iloc[-1]
            self.ui.result_last_trading_day_label.setText(last_observed_trading_day)

            upward_predictions = [prediction[0] for prediction in final_predictions if prediction[0] >= 0]
            downward_predictions = [prediction[0] for prediction in final_predictions if prediction[0] < 0]
            majority_prediction = 'Upward' if len(upward_predictions) >= len(downward_predictions) else 'Downward'

            self.ui.result_prediction_label.setText(majority_prediction)

            print(f'Stock: {stock_ticker}')

            print()
            
            print(f'Total Ups: {performances[0]["total_ups"]}')
            print(f'Total Downs: {performances[0]["total_downs"]}')

            print()

            # print average accuracies of the built models
            print(f"Mean DA: {mean_da}")
            print(f"Mean UDA: {mean_uda}")
            print(f"Mean DDA: {mean_dda}")

            print()

            print(f"Standard Dev. DA: {std_da}")
            print(f"Standard Dev. UDA: {std_uda}")
            print(f"Standard Dev. DDA: {std_dda}")

            print()

            print(f"Optimistic Baseline DA: {optimistic_baseline}")
            print(f"Pessimistic Baseline DA: {pessimistic_baseline}")

            print(f'Final Predictions: {final_predictions}')

            self.ui.status_label1.setText('Idle - Done forecasting')
            self.ui.status_label2.setText('Idle - Done forecasting')
            self.enable_ui()

        except json.decoder.JSONDecodeError:
            self.ui.status_label1.setText('[Error] Invalid stock ticker')
            self.ui.status_label2.setText('[Error] Invalid stock ticker')
            self.enable_ui()


    def clear_model_info(self):
        self.ui.last_tuning_date_label2.setText('-')
        self.ui.layers_label.setText('-')
        self.ui.units_label.setText('-')
        self.ui.dropout_label.setText('-')
        self.ui.window_size_label.setText('-')


    def update_model_info(self, hps=None):
        self.clear_model_info()
        
        if hps is None:
            last_model_tuning_date = 'Never'
            layers = 'No data'
            units = 'No data'
            dropout = 'No data'
            time_steps = 'No data'

        else:
            last_model_tuning_date = hps['last_model_tuning_date']
            layers = str(hps['layers'])
            units = str(hps['units'])
            dropout = str(hps['dropout'])
            time_steps = str(hps['time_steps'])

        self.ui.last_tuning_date_label2.setText(last_model_tuning_date)
        self.ui.layers_label.setText(layers)
        self.ui.units_label.setText(units)
        self.ui.dropout_label.setText(dropout)
        self.ui.window_size_label.setText(time_steps)


    def get_model(self):

        self.disable_ui()
        self.clear_model_info()

        try:
            stock_ticker = self.ui.stock_ticker_line_edit2.text()
            stock_ticker = stock_ticker.strip().upper()

            self.ui.stock_ticker_line_edit2.clear()

            if len(stock_ticker) == 0:
                self.ui.status_label1.setText('[Error] Invalid stock ticker')
                self.ui.status_label2.setText('[Error] Invalid stock ticker')
                self.enable_ui()
                return

            self.ui.status_label1.setText(f'Getting model information for {stock_ticker} stock...')
            self.ui.status_label2.setText(f'Getting model information for {stock_ticker} stock...')

            hps = self.get_hps(stock_ticker)

            # to ensure stock_ticker is valid (listed in PSE)
            if hps is None:
                with open('keys/EOD_API_key.txt') as file:
                    token = file.readline()
                
                get_trading_dates(stock_ticker, get_dates_five_years(testing=True), token)

            self.update_model_info(hps=hps)

            self.ui.status_label1.setText(f'Idle - Model information for {stock_ticker} displayed')
            self.ui.status_label2.setText(f'Idle - Model information for {stock_ticker} displayed')
            self.enable_ui()

        except json.decoder.JSONDecodeError:
            self.ui.status_label1.setText('[Error] Invalid stock ticker')
            self.ui.status_label2.setText('[Error] Invalid stock ticker')
            self.enable_ui()


    def tune_model(self):
        
        self.disable_ui()
        self.clear_model_info()

        try:
            stock_ticker = self.ui.stock_ticker_line_edit3.text()
            stock_ticker = stock_ticker.strip().upper()

            self.ui.stock_ticker_line_edit3.clear()

            if len(stock_ticker) == 0:
                self.ui.status_label1.setText('[Error] Invalid stock ticker')
                self.ui.status_label2.setText('[Error] Invalid stock ticker')
                self.enable_ui()
                return

            self.ui.status_label1.setText('Please check console for possible confirmation')
            self.ui.status_label2.setText('Please check console for possible confirmation')
            
            date_range = get_dates_five_years()
            hps = self.get_hps(stock_ticker)

            if (hps is None) or (date_range[1] != hps['last_model_tuning_date']):
                time_steps = 20
                _, _, train_x, train_y, _, _, _ = get_dataset(stock_ticker, date_range=date_range, time_steps=time_steps, drop_col=None)
                
                # for when user did not continue with News API historical call, will instead input new stock ticker
                if train_x is None and train_y is None:
                    self.ui.status_label1.setText('Idle - Please input new stock ticker')
                    self.ui.status_label2.setText('Idle - Please input new stock ticker')
                    self.enable_ui()
                    return

                self.ui.status_label1.setText(f'Tuning model for {stock_ticker} stock...')
                self.ui.status_label2.setText(f'Tuning model for {stock_ticker} stock...')
                
                optimal_hps = get_optimal_hps(train_x, train_y)

                os.chdir('data')

                hps_database = shelve.open('hps_database')
                hps_database_key = f"{stock_ticker}"

                if hps_database_key in hps_database:
                    hps_database.pop(hps_database_key)

                last_model_tuning_date = date_range[1]
                hps_database[hps_database_key] = {
                    'last_model_tuning_date' : last_model_tuning_date,
                    'layers' : optimal_hps['layers'],
                    'units' : optimal_hps['units'],
                    'dropout' : optimal_hps['dropout'],
                    'time_steps' : time_steps
                }

                hps = hps_database[hps_database_key]

                hps_database.close()
                os.chdir('..')

            self.update_model_info(hps=hps)

            self.ui.status_label1.setText(f'Idle - Done tuning model for {stock_ticker}')
            self.ui.status_label2.setText(f'Idle - Done tuning model for {stock_ticker}')
            self.enable_ui()

        except json.decoder.JSONDecodeError:
            self.ui.status_label1.setText('[Error] Invalid stock ticker')
            self.ui.status_label2.setText('[Error] Invalid stock ticker')
            self.enable_ui()


    def run_prediction_thread(self):
        self.worker = PredictionThread(self)
        self.worker.start()

    
    def run_getting_thread(self):
        self.worker = GettingThread(self)
        self.worker.start()

    
    def run_tuning_thread(self):
        self.worker = TuningThread(self)
        self.worker.start()


        
        





def main():

    warnings.simplefilter('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)

    print("Loading UI. Please wait.")


    app = qtw.QApplication(sys.argv)
    window = MainWindow(app)

    window.show()
    sys.exit(app.exec_())


main()
