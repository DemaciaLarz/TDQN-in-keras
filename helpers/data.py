import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

class Data:
    '''Obtains hydro data and preprocesses it.'''
        
    def data(self, test_len):
        names = ['date', 'price', 'avg_p', 'bid', 'ask',
                 'o', 'h', 'l', 'c', 'avgp', 'vol', 'oms', 'num']
        # get data
        df = pd.read_csv('pcell.csv', sep=';', header=1).iloc[:,:1]
        df[[1, 2]] = pd.read_csv('pcell.csv', sep=';', header=1).iloc[:,6:8]
        df = pd.concat([df, pd.read_csv('pcell.csv', sep=';', header=1).iloc[:,:-1].drop(
            columns=['Date'])], axis=1).iloc[::-1].reset_index().drop(columns='index')
        df.columns = names
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
            em_t = sum(df['price'][:size]) / size
            for i in range(len(df)):
                if i <= size:
                    em[size].append(0)
                else:
                    em_t = (df['price'][i] * (
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
        # slice the first 26 rows
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
        # split into train and test
        X_train = norm[:len(df4) - test_len]
        X_test = norm[len(df4) - test_len:]
        train = df4.iloc[:len(df4) - test_len]
        test = df4.iloc[len(df4) - test_len:].reset_index().drop(columns='index')
        data = df4
        return X_train, X_test, train, test, data

