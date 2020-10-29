import numpy as np
import pandas as pd
import pymongo

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import load_model

import os
import glob
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

model_1 = load_model('model_2249')
model_2 = load_model('model_5699')


class Pipeline:
    '''Provides daily results to the TradingHydro application.
    
    Propagates data from the web, via inference, to a database.
    
    Takes in some csv document and gives back numerous updates to databse.
    
    '''
    
    def __init__(self, csv_filename=False, no_scraping=False):
        '''Initializes Pipeline variables.
        
        Attributes:
        url: str, api key for mongodb.
        csv_filename: str with the csv filename.
        no_scraping: bool, True as default, False means no scraping.
        DF_out: pandas DataFrame, csv transition data.
        model_1_trans: dict with transition data.
        model_2_trans: dict with transition data.
        
        '''
        # strings
        self.url = 'your-mongodb-api-key'
        self.csv_filename = 'csv_filename.csv'
        # debugging 
        if csv_filename:
            self.csv_filename = csv_filename
        self.no_scraping = no_scraping
        # transitions
        self.DF_out = None
        self.model_1_trans = None
        self.model_2_trans = None
        
 
    def db_health_checking(self):
        '''Checks for mismatchs and doubles in the plot collections.'''
        
        # set database
        client = pymongo.MongoClient(self.url)
        db = client['powercell']

        # name collection vars so they correspond with mongodb col names
        plot_1 = db['plot_1']
        plot_2 = db['plot_2']
        plot_3 = db['plot_3']
        plot_4 = db['plot_4']

        # find the current data in respective collection 
        querys = [plot_1.find_one(), 
                  plot_3.find_one(), 
                  plot_4.find_one()]

        # clean out mongodb id object
        querys_no_id = [{i: query[i] for i in ['dates', 'lineseries']} for query in querys]

        # compare lens
        for name, query in zip(('plot_1', 'plot_3', 'plot_4'), querys_no_id):
            lens = [len(query['dates'])]
            lens = lens + [len(query['lineseries'][i]['points']) for i in range(len(query))]
            assert len(set(lens)) == 1, 'Health issue, len mismatch in plot ' + name
            
        return True
        
        
    def scraping(self):
        '''Downloads a csv file from the web to disk.
        
        Returns:
        bool, True if procedure is successful.
        
        '''
        # PREPARE FOR SCRAPE
        
        # locate yesterdays csv file in folder
        csvfiles = [file for file in glob.glob('*.csv')]
        assert len(csvfiles) == 1, 'Prep for scrape, more or less than one csv on disk.'
        
        # remove csv
        os.remove(csvfiles[0])
        assert len([file for file in glob.glob('*.csv')]) == 0, 'Remove csv, still csv on disk.'
        
        # SELENIUM
        
        # strings
        url = 'http://www.nasdaqomxnordic.com/shares/microsite?Instrument=SSE105121'
        user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 ' \
        '(KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'
        
        # options
        chrome_options = Options()
        chrome_options.add_argument('--user-agent=' + user_agent)
        chrome_options.add_argument('--headless')
        
        # download location
        download_dir = os.path.dirname(os.path.realpath('__file__'))
        prefs = {'download.default_directory' : download_dir}
        chrome_options.add_experimental_option('prefs', prefs)
        
        # wait, launch browser and wait
        time.sleep(np.random.randint(1, 120))
        driver = webdriver.Chrome(options=chrome_options)
        driver.implicitly_wait(np.random.randint(3, 15))
        
        # go to page and wait
        driver.get(url)
        driver.implicitly_wait(np.random.randint(3, 15))
        
        # find showhistory button wait and click
        show_history_class = driver.find_element_by_class_name('showHistory')
        show_history_class.click()
        driver.implicitly_wait(np.random.randint(3, 15))
        
        # find, click, download csv and wait
        exportExcel_id = driver.find_element(By.ID, 'exportExcel')
        exportExcel_id.click()
        time.sleep(5)
        
        # POST SCRAPE
        
        # change name on csv file and wait
        csvfiles = [file for file in glob.glob('*.csv')]
        assert len(csvfiles) == 1, 'Post scrape, more or less than one csv on disk.'
        os.rename(csvfiles[0], self.csv_filename)
        time.sleep(5)
        
        return True
    
        
    def preprocessing(self):
        '''Preprocess new data wrt old and slice off last.
        
        Returns:
        date: str with todays date.
        x1_s2: numpy array, x1 part of s2.
        p_t: float, yesterdays price t-1 for todays calculations.
        c_: float with todays closing price return.
        price: float, powercell raw price for today.
        ma_26: float, TA indicator.
        em_12: float, TA indicator.
        em_26: float, TA indicator.
        
        '''
        names = ['date', 'price', 'avg_p', 'bid', 'ask',
                 'o', 'h', 'l', 'c', 'avgp', 'vol', 'oms', 'num']
        
        # put scraped csv in dataframe and obtain the last row
        df_scraped = pd.read_csv(self.csv_filename, sep=';', header=1).iloc[:,:1]
        df_scraped[[1, 2]] = pd.read_csv(self.csv_filename, sep=';', header=1).iloc[:,6:8]
        df_scraped = pd.concat([df_scraped, pd.read_csv(
            self.csv_filename, sep=';', header=1).iloc[:,:-1].drop(
            columns=['Date'])], axis=1).iloc[::-1].reset_index().drop(columns='index')
        df_scraped.columns = names
        scraped_row = df_scraped.iloc[[-1]]
        
        # dataframe (DF) related database (DB) and collection
        client = pymongo.MongoClient(self.url)
        db_DF = client['DF']
        DF = db_DF['DF']
        # fetch yesterdays DF 
        df_in = db_DF.DF.find_one(sort=[('_id', pymongo.DESCENDING)])
        df_in_json = pd.read_json(df_in[list(df_in.keys())[-1]], keep_default_dates=False)
        
        # concatenate yesterdays DF and the scraped row
        df = pd.concat([df_in_json, scraped_row], axis=0).reset_index().drop(columns='index')
        # store now but update later
        self.df_out = pd.concat([df_in_json, scraped_row], axis=0).reset_index().drop(columns='index')
        # assert that the scraped row is not the same as the last in df_in
        date = df['date'].iloc[-1]
        assert date != df['date'].iloc[-2], (
            'Update Abort: scraped row is same as last. Weekend?')
        
        # Filter out null
        for name in names:
            no_null = []
            # check if null exist in column
            if any(df[name].isnull()):
                # traverse the boolean dataframe
                for i, j in enumerate(df[name].isnull()):
                    if not j:
                        # hold a value from latest non null
                        tmp = df[name].iloc[i]
                        no_null.append(tmp)
                    else:
                        no_null.append(tmp)
                # put back in dataframe
                df[name] = pd.Series(no_null)
                
        # Get float from string
        for name in names[1:]:
            if type(df[name].iloc[1]) == str:
                df[name] = pd.Series([float(i.replace(',', '.')) for i in df[name]])
  
        # Moving averages
        ma_sizes = (26,)
        ma = {i: [] for i in ma_sizes}
        for size in ma_sizes:
            for i in range(len(df)):
                if i <= size:
                    ma[size].append(np.average(df['price']))
                else:
                    value = sum(df['price'].values[i - size: i]) / size
                    ma[size].append(value)   
                    
        # Exponential moving average
        smoother = 2
        em_sizes = (12, 20, 26)
        em = {i: [] for i in em_sizes}
        for size in em_sizes:
            em_t = sum(df['price'].iloc[:size]) / size
            for i in range(len(df)):
                if i <= size:
                    em[size].append(0)
                else:
                    em_t = (df['price'].iloc[i] * (
                        smoother / (1 + size)) + (em_t * (1 - (smoother / (1 + size)))))
                    em[size].append(em_t)
                    
        # MACD
        macd1 = [i - j for i, j in zip(em[12], em[26])]
        macd2 = []
        macd3 = []
        em_t = sum(macd1[:9]) / 9
        for i in range(len(macd1)):
            if i <= 9:
                macd2.append(0)
            else:
                em_t = (macd1[i] * (
                    smoother / (1 + size)) + (em_t * (1 - (smoother / (1 + size)))))
                macd2.append(em_t)
        macd3 = [i - j for i, j in zip(macd1, macd2)]
        tech = [ma[26], em[12], em[26], macd1, macd2, macd3]
        names_df2 = ['ma1', 'em1', 'em2', 'md1', 'md2', 'md3']
        names2 = names + names_df2
        df2 = pd.DataFrame({i: j for i, j in zip(names_df2, tech)})
        # slice the first 26 rows due to moving averages
        df3 = pd.concat([df, df2], axis=1).iloc[27:]
        
        # get diff and pct change
        diff = df3[['vol', 'oms', 'num']].diff()
        pct = df3[['bid', 'ask', 'o', 'h', 'l', 'c', 'avgp'] + names_df2].pct_change()
        diff_pct = pd.concat([pct, diff], axis=1)
        diff_pct.columns = [
            name + '_' for name in [
                'bid', 'ask', 'o', 'h', 'l', 'c', 'avgp'] + names_df2 + ['vol', 'oms', 'num']]
        df4 = pd.concat([df3, diff_pct], axis=1).iloc[1:].reset_index().drop(columns='index')
        names3 = df4.columns
        
        # clipping outliers
        for name in diff_pct.columns.tolist():
            df4[[name]] = df4[[name]].clip(- 3 *df4[name].std(), 3 * df4[name].std())
            
        # Normalizing
        scaler = StandardScaler()
        norm = scaler.fit_transform(
            df4[list(diff_pct.columns)].values.reshape(-1, len(list(diff_pct.columns))))
        # Add avgp__ to df4
        df4[['avgp__']] = pd.DataFrame({None: norm[:,6:7].squeeze()})
        
        # package output
        x1_s2 = norm[-1] 
        p_t = round(float(df4['avg_p'].iloc[-2]), 3) # yesterdays price for todays calculations
        c_ = round(float(df4['c_'].iloc[-1]), 3) # todays closing price return
        price = round(float(df4['price'].iloc[-1]), 3)
        ma_26 = round(float(df4['ma1'].iloc[-1]), 3)
        em_12 = round(float(df4['em1'].iloc[-1]), 3)
        em_26 = round(float(df4['em2'].iloc[-1]), 3)
        
        return date, x1_s2, p_t, c_, price, ma_26, em_12, em_26
        
        
    def inference(self, x1_s2, p_t, c_, date): 
        '''Fetches transitions from DB, performs inference 
        and computes results. It packages the states nicely 
        for the next transitions, and finally pushes them to DB.
        
        Arguments:
        x1_s2: numpy array, x1 part of s2.
        p_t: float, yesterdays price for todays calculations.
        c_: float with todays closing price return.
        date: str with todays date.
        
        Returns:
        model_1_value: float, todays portfolio value.
        model_2_value: float, todays portfolio value.
        baseline_value: float, todays portfolio value.
        model_1_action: float, zero or one model output.
        model_2_action: float, zero or one model output.
        
        '''
        # ORIGINAL BUNDLE - ORIGINAL BUNDLE
        
        # transition related db and collections
        client = pymongo.MongoClient(self.url)
        db_transitions = client['transitions']
        trans_model_1 = db_transitions['trans_model_1']
        trans_model_2 = db_transitions['trans_model_2']
        trans_baseline = db_transitions['trans_baseline']
        
        # fetch transition components for model 1 (A)
        A_query = trans_model_1.find_one(sort=[('_id', pymongo.DESCENDING)])
        # A_s is now redundant but we let it remain
        A_s = [np.array(A_query['s'][0]).reshape(1, 16),
               np.array(A_query['s'][1]).reshape(1, 3)]
        # A_a is the action from the post calculation inference yesterday
        A_a = A_query['a'] 
        A_cash = A_query['cash']
        A_stock_v = A_query['stock_v']
        A_stock_n = A_query['stock_n']
        
        # fetch transition components for model 2 (B)
        B_query = trans_model_2.find_one(sort=[('_id', pymongo.DESCENDING)])
        # B_s is now redundant but we let it remain
        B_s = [np.array(B_query['s'][0]).reshape(1, 16),
               np.array(B_query['s'][1]).reshape(1, 3)]
        # B_a is the action from the post calculation inference yesterday
        B_a = B_query['a'] 
        B_cash = B_query['cash']
        B_stock_v = B_query['stock_v']
        B_stock_n = B_query['stock_n']
        
        # fetch transition components for baseline
        base_val_in = trans_baseline.find_one(sort=[('_id', pymongo.DESCENDING)])['value']
        
        # load models
        #model_1 = load_model('model_2249')
        #model_2 = load_model('model_5699')
        
        # NO INFERENCE pre calculations, do them post and pass the action
        
        # compute portfolio for model 1
        C = 0.02
        if A_a == 0:
            Q = np.floor(A_cash / (p_t * (1 + C))) # measure up the long position
        if A_a == 1:
            Q = -np.floor(A_stock_n) # measure up the short position
            
        A_cash = abs(A_cash - (Q * p_t) - (C * abs(Q))) # change in cash value
        A_stock_v = (A_stock_n + Q) * p_t # change in stock value
        A_stock_n = A_stock_n + Q # change in number of stock
        
        # compute portfolio for model 2
        if B_a == 0:
            Q = np.floor(B_cash / (p_t * (1 + C))) # measure up the long position
        if B_a == 1: 
            Q = -np.floor(B_stock_n) # measure up the short position
            
        B_cash = abs(B_cash - (Q * p_t) - (C * abs(Q))) # change in cash value
        B_stock_v = (B_stock_n + Q) * p_t # change in stock value
        B_stock_n = B_stock_n + Q # change in number of stock
        
        # package respective s2 states
        scaler = StandardScaler()
        
        A_x2 = scaler.fit_transform(np.array(
            [[A_cash, A_stock_v, A_stock_n]]).reshape(-1, 1)).reshape(1, 3)
        
        A_s2 = [x1_s2, A_x2]
        
        B_x2 = scaler.fit_transform(np.array(
            [[B_cash, B_stock_v, B_stock_n]]).reshape(-1, 1)).reshape(1, 3) 
        
        B_s2 = [x1_s2, B_x2]
     
        # set values
        model_1_value = A_cash + A_stock_v
        model_2_value = B_cash + B_stock_v
        baseline_value = base_val_in + (base_val_in * c_)
        
        # INFERENCE - action values for now and for calculations tomorrow.
        model_1_action = float(np.argmax(model_1.predict(A_s)))
        model_2_action = float(np.argmax(model_2.predict(B_s)))
                
        # push model 1 transition components
        trans_names = list(A_query.keys())[1:]
        self.model_1_trans = {name: val for name, val in zip(trans_names, 
                                                             [date,
                                                             [x.tolist() for x in A_s2], 
                                                             model_1_action, 
                                                             round(A_cash, 3), 
                                                             round(A_stock_v, 3), 
                                                             round(A_stock_n, 3)])}        
        # push model 2 transition components
        self.model_2_trans = {name: val for name, val in zip(trans_names, 
                                                             [date, 
                                                             [x.tolist() for x in B_s2], 
                                                             model_2_action, 
                                                             round(B_cash, 3), 
                                                             round(B_stock_v, 3), 
                                                             round(B_stock_n, 3)])}  
        
        return (round(model_1_value, 3),
                round(model_2_value, 3),
                round(baseline_value, 3),
                model_1_action, 
                model_2_action)
         
    
    def packaging_and_pushing(self, 
                              model_1_value, 
                              model_2_value, 
                              baseline_value,
                              model_1_action, 
                              model_2_action, 
                              price, 
                              ma_26, 
                              em_12, 
                              em_26, 
                              date):
        
        '''Fetches full document of plot data from DB, 
        updates it locally, then packages and pushes to DB.
        
        Arguments:
        model_1_value: float, todays portfolio value.
        model_2_value: float, todays portfolio value.
        baseline_value: float, todays portfolio value.
        model_1_action: float, zero or one model output.
        model_2_action: float, zero or one model output.
        price: float, powercell raw price for time t.
        ma_26: float, TA indicator.
        em_12: float, TA indicator.
        em_26: float, TA indicator.
        date: str, date for time t.
        
        Returns:
        bool    
        
        '''
        values = {
            'plot_1': {
                0: model_1_value,
                1: model_2_value,
                2: baseline_value},
            'plot_3': {
                0: 10000.,
                1: 10000.,
                2: 10000.,
                3: 10000.},
            'plot_4': {
                0: price,
                1: ma_26,
                2: em_12,
                3: em_26}
            }
        
        # plot_2 update 
        plot_2_update = {
            'date': date,
            'Model_1': model_1_action,
            'Model_2': model_2_action
            }
        
        # set database
        client = pymongo.MongoClient(self.url)
        db = client['powercell']
        
        # name collection vars so they correspond with mongodb col names
        plot_1 = db['plot_1']
        plot_2 = db['plot_2']
        plot_3 = db['plot_3']
        plot_4 = db['plot_4']
        
        # find the current data in respective collection 
        querys = [plot_1.find_one(), 
                  plot_3.find_one(), 
                  plot_4.find_one()]
        
        # clean out mongodb id object
        updates = [{i: query[i] for i in ['dates', 'lineseries']} for query in querys]
        
        # append date
        for update in updates:
            update['dates'].append(date)
            
        # append values
        for i in range(len(updates[0]['lineseries'])):
            updates[0]['lineseries'][i]['points'].append(values['plot_1'][i])
            updates[1]['lineseries'][i]['points'].append(values['plot_3'][i])
            updates[2]['lineseries'][i]['points'].append(values['plot_4'][i])
            
        # push to database
        res = db.plot_1.replace_one({}, updates[0])
        assert res.modified_count == 1, 'Plot 1 update to database failed.'
        res = db.plot_3.replace_one({}, updates[1])
        assert res.modified_count == 1, 'Plot 3 update to database failed.'
        res = db.plot_4.replace_one({}, updates[2])
        assert res.modified_count == 1, 'Plot 4 update to database failed.'
        # insert plot 2 update
        res = db.plot_2.insert_one(plot_2_update)
        
        # PREPROCESSING PUSHES - update DF
        # DF database
        db_DF = client['DF']
        DF = db_DF['DF']
        # push DF
        res = db_DF.DF.insert_one({date: self.df_out.to_json()})
        
        # INFERENCE PUSHES - model_1, model_2, baseline
        # transition related db and collections
        db_transitions = client['transitions']
        trans_model_1 = db_transitions['trans_model_1']
        trans_model_2 = db_transitions['trans_model_2']
        trans_baseline = db_transitions['trans_baseline']
        # push transitions
        res = db_transitions.trans_model_1.insert_one(self.model_1_trans)
        res = db_transitions.trans_model_2.insert_one(self.model_2_trans)
        res = db_transitions.trans_baseline.insert_one({'date': date, 
                                                        'value': baseline_value})
        
        return True
        
    
    def main(self):
        '''Executes the pipeline methods.'''
        
        # check database health
        self.db_health_checking()
        
         # scraping
        if not self.no_scraping:
            self.scraping()
        
        # preprocessing
        date, x1_s2, p_t, c_, price, ma_26, em_12, em_26 = self.preprocessing()
        
        # inference
        inference = self.inference(x1_s2, p_t, c_, date)
        model_1_value, model_2_value, baseline_value = inference[0], inference[1], inference[2]
        model_1_action, model_2_action = inference[3], inference[4]
        
        # packaging and pushing
        self.packaging_and_pushing(model_1_value, 
                                   model_2_value, 
                                   baseline_value, 
                                   model_1_action, 
                                   model_2_action, 
                                   price, 
                                   ma_26, 
                                   em_12, 
                                   em_26, 
                                   date)
        
        
if __name__ == '__main__':
    Pipeline().main()