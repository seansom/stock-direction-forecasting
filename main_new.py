from tkinter.messagebox import NO
from turtle import update
from tensorflow import keras, compat
from statistics import mean, stdev
import keras_tuner as kt
import numpy as np
import pandas as pd
import os, sys, math, copy, shutil, shelve, warnings, json, datetime
from sklearn.preprocessing import PowerTransformer
from data_processing_new import get_dataset, inverse_transform_data, get_dates_five_years, get_trading_dates, get_final_window, scale_transform_window
from direction_forecasting_new import CustomCallback, make_lstm_hypermodel, get_optimal_hps, make_lstm_model, forecast_lstm_model, get_lstm_model_perf, print_model_performance, experiment, make_forecast
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


    def get_params(self, stock_ticker):

        os.chdir('data')

        params_database = shelve.open('params_database')
        params_database_key = f"{stock_ticker}"

        if params_database_key in params_database:
            params = params_database[params_database_key]

        else:
            params = {
                'time_steps': 1,
                'hps': {
                    'layers': 3,
                    'units': 64,
                    'dropout': 0.6
                },
                'dropped_features': None,
                'last_model_tuning_date': 'Never'
            }

        params_database.close()
        os.chdir('..')

        return params


    def get_model_info(self, stock_ticker, last_model_tuning_date):

        os.chdir('data')

        model_info_database = shelve.open('model_info_database')
        model_info_database_key = f"{stock_ticker}"

        if model_info_database_key in model_info_database and last_model_tuning_date in [model_info_database[model_info_database_key]['training_date'], 'Never']:
            model_info = model_info_database[model_info_database_key]

        else:
            model_info = None

        model_info_database.close()
        os.chdir('..')

        return model_info


    def make_models(self, stock_ticker, params):

        time_steps = params['time_steps']
        hps = params['hps']
        dropped_features = params['dropped_features']
        last_model_tuning_date = params['last_model_tuning_date']

        if last_model_tuning_date != 'Never':
            date_five_years_ago = datetime.datetime.strptime(last_model_tuning_date, '%Y-%m-%d') - datetime.timedelta(days=round(365.25 * 5))
            date_range = (date_five_years_ago, last_model_tuning_date)
        
        else:
            date_range = ('2017-04-13', '2022-04-13')

        performances = []
        repeats = 2

        self.ui.models_progress_label.setText(f'0 / {repeats}')
        print("===================================================")
        for i in range(repeats):
            print(f"Experiment {i + 1} / {repeats}")

            self.ui.status_label1.setText(f'Building {stock_ticker} model {i + 1}...')
            self.ui.status_label2.setText(f'Building {stock_ticker} model {i + 1}...')
            perf, model, linear_scaler, scaler, col_names = experiment(stock_ticker, time_steps, date_range=date_range, drop_col=dropped_features, hps=hps, window=self)

            performances.append(perf)

            self.ui.status_label1.setText(f'Saving {stock_ticker} model {i + 1}...')
            self.ui.status_label2.setText(f'Saving {stock_ticker} model {i + 1}...')
            os.chdir('models')
            model.save(f'{stock_ticker} model {i + 1}', overwrite=True)
            os.chdir('..')


            print("===================================================")
            self.ui.models_progress_label.setText(f'{i + 1} / {repeats}')



        mean_da = round(mean([perf['da'] for perf in performances]), 6)
        mean_uda = round(mean([perf['uda'] for perf in performances]), 6)
        mean_dda = round(mean([perf['dda'] for perf in performances]), 6)

        std_da = round(stdev([perf['da'] for perf in performances]), 6)
        std_uda = round(stdev([perf['uda'] for perf in performances]), 6)
        std_dda = round(stdev([perf['dda'] for perf in performances]), 6)

        total_ups = performances[0]['total_ups']
        total_downs = performances[0]['total_downs']
        
        optimistic_baseline = round(total_ups / (total_ups + total_downs), 6)
        pessimistic_baseline = round(1 - optimistic_baseline, 6)


        os.chdir('data')

        model_info_database = shelve.open('model_info_database')
        model_info_database_key = f"{stock_ticker}"

        model_info_database[model_info_database_key] = {
            'linear_scaler': linear_scaler,
            'scaler': scaler,
            'col_names': col_names,
            'training_date': date_range[1],
            'mean_da': mean_da,
            'mean_uda': mean_uda,
            'mean_dda': mean_dda,
            'std_da': std_da,
            'std_uda': std_uda,
            'std_dda': std_dda,
            'optimistic_baseline': optimistic_baseline,
            'pessimistic_baseline': pessimistic_baseline
        }

        model_info_database.close()
        os.chdir('..')
        return


    def make_prediction(self):

        repeats = 2

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

            performances = []
            final_predictions = []

            self.ui.status_label1.setText(f'Getting {stock_ticker} model parameters...')
            self.ui.status_label2.setText(f'Getting {stock_ticker} model parameters...')

            params = self.get_params(stock_ticker)

            time_steps = params['time_steps']
            hps = params['hps']
            dropped_features = params['dropped_features']
            last_model_tuning_date = params['last_model_tuning_date']


            self.ui.status_label1.setText(f'Checking for {stock_ticker} models...')
            self.ui.status_label2.setText(f'Checking for {stock_ticker} models...')

            model_info = self.get_model_info(stock_ticker, last_model_tuning_date)
            if model_info is None:
                self.make_models(stock_ticker, params)
                model_info = self.get_model_info(stock_ticker, last_model_tuning_date)

            else:
                self.ui.train_progress_label.setText('100.0 %')
                self.ui.prediction_progress_label.setText('100.0 %')
                self.ui.models_progress_label.setText(f'{repeats} / {repeats}')


            self.ui.last_tuning_date_label1.setText(last_model_tuning_date)
            self.ui.mean_da_label.setText(str(model_info['mean_da']))
            self.ui.mean_uda_label.setText(str(model_info['mean_uda']))
            self.ui.mean_dda_label.setText(str(model_info['mean_dda']))
            self.ui.stdev_da_label.setText(str(model_info['std_da']))
            self.ui.stdev_uda_label.setText(str(model_info['std_uda']))
            self.ui.stdev_dda_label.setText(str(model_info['std_dda']))

            self.ui.result_stock_ticker_label.setText(stock_ticker)


            forecasts = []

            for i in range(repeats):

                self.ui.status_label1.setText(f'Forecasting {stock_ticker} ({i + 1} / {repeats})...')
                self.ui.status_label2.setText(f'Forecasting {stock_ticker} ({i + 1} / {repeats})...')

                os.chdir('models')
                model = keras.models.load_model(f'{stock_ticker} model {i + 1}')
                os.chdir('..')

                model_dict = {
                    'model': model,
                    'linear_scaler': model_info['linear_scaler'],
                    'scaler': model_info['scaler'],
                    'col_names': model_info['col_names'],
                    'params': params
                }

                forecast, last_observed_trading_day = make_forecast(stock_ticker, model_dict)
                forecasts.append(forecast)

            print(forecasts)

            self.ui.status_label1.setText('Idle')
            self.ui.status_label2.setText('Idle')
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
