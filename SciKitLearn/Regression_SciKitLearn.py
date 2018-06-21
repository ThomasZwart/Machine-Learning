import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle 

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
# Pak alle nuttige features
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# Zodat je later kan aanpassen
forecast_col = 'Adj. Close'

# Alle NA worden outliers, want de classifier kan deze niet aan
df.fillna(-99999, inplace=True)

# Hoever in de toekomst je wilt voorspellen
forecast_out = int(math.ceil(0.1*len(df)))

# Shift de data, zodat wat je wilt voorspellen nu tegen de features staat
# a.k.a supervised learning
df['label'] = df[forecast_col].shift(-forecast_out)
# maakt een array van arrays van de features
X = np.array(df.drop(['label'], 1))
# elke waarde wordt nu x = (x - x.mean)/(standard dev x)
# a.k.a hoeveel standaard deviaties x van de mean af zit
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# door de shift hebben sommige rows geen labels, die gaan nu weg
df.dropna(inplace=True)
y = np.array(df['label'])

# husselt alle sets en maakt deze 4 aan
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs = -1)
clf.fit(X_train, y_train)
with open('linearregression2.pickle', 'wb') as file:
    pickle.dump(clf, file)

# De score is (1 - u/v), waarin u = som van errors = sum((y_true - y_predic)^2)
# en v = sum((y_true - y_true_mean)^2), zie https://en.wikipedia.org/wiki/Coefficient_of_determination
# de accuracy is heel hoog, want de variantie van y_test is te hoog, de data is
# uit 2005, en y_test is gehusselt door model_selection hierboven.
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

# gare grafiek
df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
