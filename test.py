all_features = ['log_return', 'ad', 'wr', 'cmf', 'atr', 'rsi', 'cci', 'adx', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment']

dropped_features = ['wr', 'cmf', 'atr', 'rsi', 'cci', 'adx', 'slope', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment']

features = [f for f in all_features if f not in dropped_features]

print(features)