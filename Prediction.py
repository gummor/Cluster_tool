import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    RocCurveDisplay,
    confusion_matrix
)

# Import data
data = pd.read_csv(loren)

scaler = RobustScaler()
data['N_normalized'] = scaler.fit_transform(data[['N']])
data['Iq1_normalized'] = scaler.fit_transform(data[['Iq1']])
data['Iq3_normalized'] = scaler.fit_transform(data[['Iq3']])
data['Copeptin_normalized'] = scaler.fit_transform(data[['Copeptin']])

data['cv'] = (data['Iq3_normalized'] - data['Iq1_normalized']) / data['Copeptin_normalized']

# Split the data into training and testing sets
X = data[['cv']]
y = data['Group']
N = data['N_normalized'] 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, N_train, N_test = train_test_split(
    X, y, N, test_size=0.33, random_state=42
)

classifiers = {
    'Random Forest': RandomForestClassifier(
        n_estimators=20,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1, 
        class_weight='balanced', 
        random_state=42
    ),
    'Extra Trees': ExtraTreesClassifier(
        n_estimators=20,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1, 
        class_weight='balanced',
        random_state=42
    ),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=200,
        activation='logistic',
        solver='lbfgs',
        learning_rate='constant',
        max_iter=9000,
        random_state=42
    )
}

# Quality control
def train_and_evaluate(classifiers, X_train, y_train, N_train, X_test, y_test):
    results = {}
    
    for name, clf in classifiers.items():
        # Train the classifier with or without sample weights
        if name == 'Neural Network':
            clf.fit(X_train, y_train)
        else:
            clf.fit(X_train, y_train, sample_weight=N_train)

        y_pred = clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[name] = {
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1_score': f1,
            'classification_report': report
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}")
        
        cm = confusion_matrix(y_test, y_pred)
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.title(f'Matriz de Confusão - {name}')
        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')

        plt.subplot(1, 2, 2)
        RocCurveDisplay.from_estimator(clf, X_test, y_test)
        plt.title(f'Curva ROC - {name}')
    plt.tight_layout() 
    plt.show()
    
    return results

results = train_and_evaluate(classifiers, X_train, y_train, N_train, X_test, y_test)