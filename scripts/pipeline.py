from kaggle_functions_pipeline import *

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from xgboost import XGBRegressor


######## Loading the datasets ##########

train_file_path = "../data/train.parquet"
train = pd.read_parquet(train_file_path).drop(columns="bike_count")

test_file_path = "../data/final_test.parquet"
test = pd.read_parquet(test_file_path)

weather_file_path = "../external_data/external_data.csv"
weather = pd.read_csv(weather_file_path)

##################################################################################################


######## Defining the pipeline functions ##########

vacation_encoder = FunctionTransformer(encode_vacation, validate=False)
holidays_encoder = FunctionTransformer(encode_holidays, validate=False)
lockdown_curfew_encoder = FunctionTransformer(encode_lockdown_curfew, validate=False)
weather_merger = FunctionTransformer(lambda df: merge_weather(df, weather), validate=False)
column_dropper = FunctionTransformer(drop_columns, validate=False)
date_encoder = FunctionTransformer(encode_dates, validate=False)

#########################################################


date_cols = encode_dates(train[["date"]]).columns.tolist()

preprocessor = ColumnTransformer(
    [
        ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
    ],
    remainder="passthrough"
)


pipeline = Pipeline(
    [
        ("dropper", column_dropper),
        ("vacation", vacation_encoder),
        ("holiday", holidays_encoder),
        ("lockdown curfew", lockdown_curfew_encoder),
        ("weather merger", weather_merger),
        ("date", date_encoder),
        ("preprocessor", preprocessor),
        ("linear_regressor", XGBRegressor(n_estimators=10, 
                                          learning_rate=0.1, 
                                          max_depth=20, 
                                          random_state=42, 
                                          objective='reg:squarederror'))
    ]
)

X = train.drop(columns=["log_bike_count"])
y = train["log_bike_count"]

pred_file_path = "../prediction_csvs/final_test_sub.csv"
kaggle_prediction(pipeline.fit(X, y), test, pred_file_path)