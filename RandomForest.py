from sklearn.ensemble import RandomForestClassifier
from Data_Preprocessing import X_train, X_val, y_train, y_val
from Evaluate import evaluate

model = RandomForestClassifier(
    n_estimators=100,    
    max_depth=7,          
    random_state=42,
    n_jobs=-1           
)

model.fit(X_train, y_train)

evaluate(model, X_val, y_val, "Random Forest")