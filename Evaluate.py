from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

def evaluate(model, X_val, y_val, name="Model"):
    y_pred = model.predict(X_val)

    # Nếu model có predict_proba thì dùng, không thì dùng decision_function
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val)[:, 1]
    else:
        y_prob = model.decision_function(X_val)

    acc = accuracy_score(y_val, y_pred)
    pre = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)

    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC AUC : {auc:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
