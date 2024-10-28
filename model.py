import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


#reading and cleaning
df = pd.read_csv('train.csv')
df = df.dropna(subset = 'positive')
test_ = pd.read_csv('test_.csv')
test_ = test_.dropna(subset = 'positive')

#creatign a pandas dataframe
train_X = df[['purchase_price_x', 'rating', 'sentiment','positive', 'negative', 'neutral']]
train_y = df['cause']
train_y = pd.get_dummies(train_y, dtype = int)
cause_1 = train_y.BAD_QUALITY
cause_2 = train_y.DEFECTED
cause_3 = train_y.PHOTO_MISMATCH
cause_4 = train_y.WRONG_ITEM
cause_5 = train_y.WRONG_SIZE

#initializing the Regressors
model_1 = RandomForestRegressor(random_state=1)
model_2 = RandomForestRegressor(random_state=1)
model_3 = RandomForestRegressor(random_state=1)
model_4 = RandomForestRegressor(random_state=1)
model_5 = RandomForestRegressor(random_state=1)

#fitting the models
model_1.fit(train_X, cause_1)
model_2.fit(train_X, cause_2)
model_3.fit(train_X, cause_3)
model_4.fit(train_X, cause_4)
model_5.fit(train_X, cause_5)

#predicting the results
test_X = test_[['purchase_price_x', 'rating', 'sentiment','positive', 'negative', 'neutral']]
cause_pred = pd.DataFrame({'DEFECTED':model_2.predict(test_X),'WRONG_ITEM': model_4.predict(test_X),
                           'BAD_QUALITY': model_1.predict(test_X), 'PHOTO_MISMATCH':model_3.predict(test_X),
                           'WRONG_SIZE':model_5.predict(test_X)})

final = test_[['product_id', 'order_item_id']].join(cause_pred)

#svaing the results
csv_file_path = 'results.csv'
final.to_csv(csv_file_path, index=True)