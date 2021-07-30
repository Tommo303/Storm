import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix#, accuracy_score#, mean_squared_error
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import requests
import pickle
import math
import json
from pathlib import Path


# Read data from .csv to dataframe
file_path = 'C:/Users/tmorton/OneDrive - Catholic Education Western Australia/Other/Young ICT Explorers/STORM 2020/Program/cyclone_data/2019079S15120.csv'
df = pd.read_csv(file_path)

# Set dependent and independent variables
X1, y1 = df[['USA WIND', 'USA PRES', 'STORM SPEED']], df[['STORM SPEED']]
X1['STORM SPEED'] = X1['STORM SPEED'].shift(-1)

X2, y2 = df[['USA WIND', 'USA PRES', 'STORM DIR']], df[['STORM DIR']]
X2['STORM DIR'] = X2['STORM DIR'].shift(-1)


# Predicting the speed of the cyclone at the next data point
def PredictSpeed(X, y, data):
    # If there exists a saved model use it, otherwise train new model
    my_file = Path('C:/Users/tmorton/OneDrive - Catholic Education Western Australia/Other/Young ICT Explorers/STORM 2020/Program/speed.pickle.dat')
    
    if not my_file.is_file():    
        # Split data into training and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        
        # Converting the dataframe into XGBoost’s Dmatrix object
        dtrain = xgb.DMatrix(X_test, label=y_test)
        
        # Bayesian Optimization function for xgboost
        # specify the parameters you want to tune as keyword arguments
        def bo_tune_xgb(max_depth, gamma, n_estimators, learning_rate):
            params = {'max_depth': int(max_depth),
                      'gamma': gamma,
                      'n_estimators': int(n_estimators),
                      'learning_rate':learning_rate,
                      'subsample': 0.8,
                      'eta': 0.1,
                      'eval_metric': 'rmse'}
            # Cross validating with the specified parameters in 5 folds and 70 iterations
            cv_result = xgb.cv(params, dtrain, num_boost_round=70, nfold=5) #(70, 5)
            # Return the negative RMSE
            return -1.0 * cv_result['test-rmse-mean'].iloc[-1]
        
        # Invoking the Bayesian Optimizer with the specified parameters to tune
        xgb_bo = BayesianOptimization(bo_tune_xgb, {'max_depth': (1, 10), #(1, 10)
                                                     'gamma': (0, 1), #(0, 1)
                                                     'learning_rate':(0, 1), #(0, 1)
                                                     'n_estimators':(100, 120) #(100, 120)
                                                    })
        
        # Performing Bayesian optimization for 5 iterations with 8 steps of random
        # exploration with an acquisition function of expected improvement
        xgb_bo.maximize(n_iter=25, init_points=40, acq='ei') #(5, 8)
        
        
        # Extracting the best parameters 
        params = xgb_bo.max['params']
                    
        # Converting the max_depth and n_estimator values from float to int
        params['max_depth'] = int(params['max_depth'])
        params['n_estimators'] = int(params['n_estimators'])
        
        # Initialize an XGBClassifier with the tuned parameters and fit the training data
        cl = XGBClassifier(**params).fit(X_test, y_test)
        
        # Predicting for training set
        preds = cl.predict(X_test)
        
        # Graph prediction and real
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.plot(df['ISO_TIME_________'][-(y_test.shape[0]):], y_test, color='r')
        ax.plot(df['ISO_TIME_________'][-(preds.shape[0]):], preds, color='b')
        ax.set_xlabel('Date')
        ax.set_ylabel('Storm Speed')
        ax.set_title('XGBoost: Storm Speed (Actual vs Predicted)')
        plt.show()
        
        # Save model
        pickle.dump(cl, open('C:/Users/tmorton/OneDrive - Catholic Education Western Australia/Other/Young ICT Explorers/STORM 2020/Program/speed.pickle.dat', 'wb'))
    
    
    # Load saved model
    loaded_model = pickle.load(open('C:/Users/tmorton/OneDrive - Catholic Education Western Australia/Other/Young ICT Explorers/STORM 2020/Program/speed.pickle.dat', 'rb')) 
    
    # Predict on current data
    future_preds = loaded_model.predict(data[['USA WIND', 'USA PRES', 'STORM SPEED']])
    
    return future_preds
    
# Predicting the direction of the cyclone at the next data point
def PredictDirection(X, y, data):
    # If there exists a saved model use it, otherwise train new model
    my_file = Path('C:/Users/tmorton/OneDrive - Catholic Education Western Australia/Other/Young ICT Explorers/STORM 2020/Program/direction.pickle.dat')
    
    if not my_file.is_file(): 
        # Split data into training and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
        
        # Initialise XGBoost classifier and fit to training set
        cl = XGBClassifier(n_estimators=100, seed=123).fit(X_train, y_train)
        
        # Save model
        pickle.dump(cl, open('C:/Users/tmorton/OneDrive - Catholic Education Western Australia/Other/Young ICT Explorers/STORM 2020/Program/direction.pickle.dat', 'wb'))
        
        # Predicting for test set
        preds = cl.predict(X_test)
        
        # Graph prediction and real
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.plot(df['ISO_TIME_________'][-(y_test.shape[0]):], y_test, color='r')
        ax.plot(df['ISO_TIME_________'][-(preds.shape[0]):], preds, color='b')
        ax.set_xlabel('Date')
        ax.set_ylabel('Storm Direction')
        ax.set_title('XGBoost: Storm Direction (Actual vs Predicted)')
        plt.show()
        
    # Load saved model
    loaded_model = pickle.load(open('C:/Users/tmorton/OneDrive - Catholic Education Western Australia/Other/Young ICT Explorers/STORM 2020/Program/direction.pickle.dat', 'rb')) 
    
    # Predict on current data
    future_preds = loaded_model.predict(data[['USA WIND', 'USA PRES', 'STORM DIR']])
    
    return future_preds

# Retrieve current data on active cyclones
def GetCurrent():
    # TideTech - Get current cyclone data
    tt_url1 = 'https://storm.tidetech.org/v1/active'
    
    response = requests.get(tt_url1, verify=False)
    
    active_cyclones = response.json()
    
    print(active_cyclones)
    
    tt_data = pd.DataFrame(active_cyclones['data'])
    
    for i in range(len(active_cyclones['data'])):
        tt_url2 = tt_data['details'][i]
        
        response = requests.get(tt_url2, verify=False)
        
        tt_details = response.json()
        
        tt_data['details'][i] = [tt_details]
        
        
        # WeatherAPI - Get current weather data (in vicinity of cyclone)
          
        api_key = 'ceb58c04cf43451c9a620511200909'
          
        lat = tt_data['details'][i][0]['data']['position'][0]
        lon = tt_data['details'][i][0]['data']['position'][1]
        
        w_url = 'https://api.weatherapi.com/v1/current.json?key=' + api_key + '&q=' + str(lat) + ',' + str(lon)   
    
        response = requests.get(w_url, verify=False) 
       
        weather_data = response.json() 
        
        tt_data['details'][i].append([weather_data['current']['wind_kph']/1.852, 
                                      weather_data['current']['pressure_mb']])
        
    return tt_data
    
            
# Predict 'Storm Speed' and 'Storm Direction'
current = GetCurrent()

output = {}

for i in range(len(current)):
    data = {'STORM SPEED': current['details'][i][0]['data']['movement']['KTS'], 
            'STORM DIR': current['details'][i][0]['data']['movement']['bearing'], 
            'USA WIND': current['details'][i][1][0],
            'USA PRES': current['details'][i][1][1]}
    
    lon = current['details'][i][0]['data']['position'][0]
    lat = current['details'][i][0]['data']['position'][1]

    for j in range(1, 121):
        data_df = pd.DataFrame.from_dict([data])
        
        pred_speed = PredictSpeed(X1, y1, data_df)
        pred_dir = PredictDirection(X2, y2, data_df)  
            
        # Calculate next position based on predicted velocity vector
        s = (data['STORM SPEED']*1.852 + pred_speed*1.852) / 2
         
        d = (data['STORM DIR']-270 + pred_dir-270) / 2
        d = -40
        dis_v = data['STORM SPEED'] * math.sin(math.radians(d)) * 1000 / 3600
        dis_h = data['STORM SPEED'] * math.cos(math.radians(d)) * 1000 /3600
        lon = lon + dis_v / 111.12
        lat = lat + dis_h / 111.12
        
        if j % 40 == 0:
            storm_name = current['name'][i]
            
            if storm_name not in list(output.keys()):
                output[storm_name] = [[lat, lon]]
            else:
                output[storm_name].append([lat, lon])
            
            data['STORM SPEED'] = float(pred_speed)
            data['STORM DIR'] = float(pred_dir)
            
            print(pred_speed, pred_dir, ' ', s, d)
 

# Convert output to json
output = json.dumps(output)

# Update github file with predictions
from github import Github

username = "DarkSimilarity"
password = "Lamborghini1200"

# authenticate to github
g = Github('c94f59647b8f0f6e5872a98ef392429118d12556')

# get the authenticated user
repo = g.get_user().get_repo('storm')

contents = repo.get_contents('storm.json')

repo.update_file(contents.path, '', output, contents.sha)

    
   
    
    
