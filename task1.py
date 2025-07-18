import pandas as pan
import sklearn.linear_model as skl
train = pan.read_csv('train.csv')
test = pan.read_csv('test.csv')
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
target = 'SalePrice'

Xtrain = train[features]
Ytrain = train[target]
Xtest = test[features]

Xtest = Xtest.fillna(Xtrain.mean())

model = skl.LinearRegression()
model.fit(Xtrain,Ytrain)
results = model.predict(Xtest)

sub = pan.DataFrame({
    'Id' : test['Id'],
    'SalePrice' : results
})
sub.to_csv('Final result.csv',index=False)
print('predictions done')
