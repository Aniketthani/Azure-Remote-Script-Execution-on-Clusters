from sklearn.ensemble import RandomForestClassifier
from azureml.core import Workspace
from azureml.core.run import Run, _OfflineRun
import argparse
import os
import joblib
from sklearn.model_selection import train_test_split

run = Run.get_context()
ws = None
if type(run) == _OfflineRun:
    ws = Workspace.from_config()
else:
    ws = run.experiment.workspace

diabetes_ds = ws.datasets['diabetes']
diabetes_df=diabetes_ds.to_pandas_dataframe()
X_train,X_val,y_train,y_val=train_test_split(diabetes_df.drop(["Outcome"],axis=1),diabetes_df["Outcome"],test_size=0.2)

def train_and_evaluate(run, n_est, X_t, y_t, X_v, y_v):
  model = RandomForestClassifier(n_estimators=n_est)
  model.fit(X_t, y_t)
  predictions = model.predict(X_v)
  

  run.log("n_estimators", n_est)
  
  return model

model = train_and_evaluate(run, n_est=500,
                X_t=X_train,y_t= y_train,X_v= X_val, y_v=y_val)

os.makedirs('./outputs', exist_ok=True)
model_file_name = 'model.pkl'
joblib.dump(value=model,
            filename=
                  os.path.join('./outputs/',model_file_name))