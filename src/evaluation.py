import numpy as np
from sklearn.metrics import roc_auc_score, mean_absolute_error

def evaluate_model(model, X_val, y_cls_val, y_reg_val):

    print(f"\n[5/6] Evaluating model...")
    
    predictions = model.predict(X_val)
    cls_predictions = predictions[0].flatten()  
    reg_predictions = predictions[1].flatten()  

    auc = roc_auc_score(y_cls_val, cls_predictions)
    mae = mean_absolute_error(y_reg_val, reg_predictions)

    print(f"Classification AUC: {auc:.4f}")
    print(f"Regression MAE: {mae:.2f} minutes")

    return cls_predictions, reg_predictions, auc, mae