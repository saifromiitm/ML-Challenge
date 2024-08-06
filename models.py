import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

file_path = "C:\Users\Dell\Downloads\labelled_train_set.csv"
data = pd.read_csv(file_path)

X = data['Article']
y = data['Type'].apply(lambda x: 1 if x == 'AI-generated' else 0)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42, stratify=y)

def tune_evaluate_model(model, param_grid, X_train, y_train, X_test, y_test):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return accuracy, report, best_model


log_reg = LogisticRegression()
log_reg_param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
log_reg_acc, log_reg_report, best_log_reg = tune_evaluate_model(log_reg, log_reg_param_grid, X_train, y_train, X_test, y_test)

naive_bayes = MultinomialNB()
nb_param_grid = {
    'alpha': [0.1, 0.5, 1, 5, 10]
}
nb_acc, nb_report, best_naive_bayes = tune_evaluate_model(naive_bayes, nb_param_grid, X_train, y_train, X_test, y_test)


decision_tree = DecisionTreeClassifier()
dt_param_grid = {
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt_acc, dt_report, best_decision_tree = tune_evaluate_model(decision_tree, dt_param_grid, X_train, y_train, X_test, y_test)

print("Best Logistic Regression Model:\n", best_log_reg)
print("Logistic Regression Report:\n", log_reg_report)
print("Logistic Regression Accuracy:", log_reg_acc)

print("Best Naive Bayes Model:\n", best_naive_bayes)
print("Naive Bayes Report:\n", nb_report)
print("Naive Bayes Accuracy:", nb_acc)

print("Best Decision Tree Model:\n", best_decision_tree)
print("Decision Tree Report:\n", dt_report)
print("Decision Tree Accuracy:", dt_acc)


content = """
Netnews, also known as Usenet, is a global network of discussion forums, where people can post and read messages on various topics. 
It was developed in the late 1970s as a way for users of different computer systems to share information and communicate with each other. 
Netnews uses a distributed architecture, where messages are stored on servers called Usenet hosts, and are replicated across multiple hosts to ensure redundancy and availability. 
Users can access netnews using specialized newsreader software, which allows them to browse and participate in discussions on various newsgroups. 
While netnews has declined in popularity since the rise of the World Wide Web and other social media platforms, it still remains an important part of the internet's history and is used by a dedicated community of users today.
"""

content_vectorized = vectorizer.transform([content])

log_reg_prediction = best_log_reg.predict(content_vectorized)
nb_prediction = best_naive_bayes.predict(content_vectorized)
dt_prediction = best_decision_tree.predict(content_vectorized)

def convert_prediction(prediction):
    return "AI-generated" if prediction == 1 else "Human-written"

log_reg_label = convert_prediction(log_reg_prediction[0])
nb_label = convert_prediction(nb_prediction[0])
dt_label = convert_prediction(dt_prediction[0])


print("Logistic Regression Prediction:", log_reg_label)
print("Naive Bayes Prediction:", nb_label)
print("Decision Tree Prediction:", dt_label)



logistic_regression_model_path = 'logistic_regression_model.pkl'
naive_bayes_model_path = 'naive_bayes_model.pkl'
decision_tree_model_path = 'decision_tree_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'

'''Pickle the models'''

with open(logistic_regression_model_path, 'wb') as file:
    pickle.dump(best_log_reg, file)


with open(naive_bayes_model_path, 'wb') as file:
    pickle.dump(best_naive_bayes, file)

with open(decision_tree_model_path, 'wb') as file:
    pickle.dump(best_decision_tree, file)

with open(vectorizer_path, 'wb') as file:
    pickle.dump(vectorizer, file)







