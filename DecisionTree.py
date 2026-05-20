from sklearn.tree import DecisionTreeClassifier
from Data_Preprocessing import X_train, X_val, y_train, y_val
from Evaluate import evaluate


# Model
model = DecisionTreeClassifier(
    max_depth=7,         
    random_state=42
)


model.fit(X_train, y_train)

# Đánh giá
evaluate(model, X_val, y_val, "Decision Tree")