import pandas as pd
import numpy as np


### Function to import the predictions as a CSV file ### 

def kaggle_prediction(fitted_model, test_set, file_path):
    """
    Return a .csv file for kaggle submission (predictions of the test dataset)
    Parameters:
        - model : a fitted sklearn model object
        - test_set : the dataset to predict log_bike_count on 
    Output:
        - file.csv : a .csv file to submit to kaggle 
    """

    y_pred = fitted_model.predict(test_set)
    y_pred_df = pd.DataFrame(y_pred, columns=["log_bike_count"])
    y_pred_df.index.name = "Id"

    y_pred_df.to_csv(file_path)

    return None


### DATA PREPARATION AND ENRICHMENT ###

## Adding vacation dates, holidays and curfews (COVID pandemic) ##

# Vacations in Paris (Zone C) for the academic year 2020-2021 # 
vacation_paris_2020 = [
    ("2020-10-17", "2020-11-01"),  
    ("2020-12-19", "2021-01-03"),  
    ("2021-02-13", "2021-02-28"),  
    ("2021-04-17", "2021-05-02"), 
    ("2021-07-06", "2021-08-31"), 
]
vacation_paris_2020 = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in vacation_paris_2020]

def encode_vacation(X, vacation=vacation_paris_2020):
    X = X.copy()
    X["is_vacation"] = 0
    for start, end in vacation:
        X["is_vacation"] |= (X["date"] >= start) & (X["date"] <= end)
    X["is_vacation"] = X["is_vacation"].astype(int)
    return X


# National Holidays in France #
holidays = [
    "2020-11-01",  
    "2020-11-11", 
    "2020-12-25",  
    "2021-01-01",  
    "2021-04-05",  
    "2021-05-01",  
    "2021-05-08",  
    "2021-05-13", 
    "2021-05-24",  
    "2021-07-14",  
    "2021-08-15", 
]
holidays = pd.to_datetime(holidays)


def encode_holidays(X, holidays=holidays):
    X = X.copy()
    X["holidays"] = X["date"].apply(lambda x: 1 if x in holidays else 0)
    return X


# Lockdowns and curfews due to the COVID-19 Pandemic # 

vacation_paris_2020 = [
    ("2020-10-17", "2020-11-01"),  
    ("2020-12-19", "2021-01-03"),  
    ("2021-02-13", "2021-02-28"),  
    ("2021-04-17", "2021-05-02"), 
    ("2021-07-06", "2021-08-31"), 
]

vacation_paris_2020 = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in vacation_paris_2020]

def encode_vacation(X, vacation=vacation_paris_2020):
    X = X.copy()
    X["is_vacation"] = 0
    for start, end in vacation:
        X["is_vacation"] |= (X["date"] >= start) & (X["date"] <= end)
    X["is_vacation"] = X["is_vacation"].astype(int)

    return X



holidays = [
    "2020-11-01",  
    "2020-11-11", 
    "2020-12-25",  
    "2021-01-01",  
    "2021-04-05",  
    "2021-05-01",  
    "2021-05-08",  
    "2021-05-13", 
    "2021-05-24",  
    "2021-07-14",  
    "2021-08-15", 
]
holidays = pd.to_datetime(holidays)


lockdown = [
    ("2020-10-30", "2020-12-15", "lockdown"),
    ("2021-03-20", "2021-04-17", "lockdown"), 
]

curfew = [
    ("2020-10-17", "2020-12-14", "21:00", "06:00", "curfew"),
    ("2020-12-15", "2021-01-15", "20:00", "06:00", "curfew"),
    ("2021-01-16", "2021-03-19", "18:00", "06:00", "curfew"),
    ("2021-03-20", "2021-06-08", "19:00", "06:00", "curfew"),
    ("2021-06-09", "2021-06-19", "23:00", "06:00", "curfew"),
]

lockdown = [(pd.to_datetime(start), pd.to_datetime(end), label) for start, end, label in lockdown]
curfew = [(pd.to_datetime(start), pd.to_datetime(end), start_hour, end_hour, label) 
          for start, end, start_hour, end_hour, label in curfew]

def encode_lockdown_curfew(X, lockdown=lockdown, curfew=curfew):
    X = X.copy()
    X["is_lockdown"] = 0
    X["is_curfew"] = 0

    for start, end, label in lockdown:
        X["is_lockdown"] |= (X["date"] >= start) & (X["date"] <= end)

    for start, end, start_hour, end_hour, label in curfew:
        X["is_curfew"] |= ((X["date"] >= start) & (X["date"] <= end) & 
                           ((X["date"].dt.time >= pd.to_datetime(start_hour).time()) | 
                            (X["date"].dt.time <= pd.to_datetime(end_hour).time())))
    return X

## Data Enrichment: Adding external data ##

weather_columns = ["u", "ff", "rr1", "t"]
columns_to_merge = weather_columns + ["date"]


def merge_weather(X, weather, columns_to_merge=columns_to_merge):
    weather = weather.copy()
    weather = weather[columns_to_merge]
    weather["date"] = weather["date"].astype("datetime64[us]")
    
    grouped_weather = weather.groupby("date").mean()

    X = X.copy()
    
    # Perform the merge while keeping the left index
    merged_df = pd.merge(X, grouped_weather, on="date", how="left")

    return merged_df


## Data Manipulation: Cleaning the Dataset ##

columns_to_drop = ["counter_technical_id", "counter_name", "site_name", "coordinates", "site_id", "counter_id",
                         "counter_installation_date"]

def drop_columns(X, columns_to_drop=columns_to_drop):
    return X.drop(columns=columns_to_drop)

def encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the "date" columns
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])