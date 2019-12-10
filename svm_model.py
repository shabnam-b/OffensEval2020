from utils.utils import load_train_data, load_test_data, clean_tweets, build_word_dict, load_glove_embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

vectorizer = TfidfVectorizer()
x_a, y_a, y_b = load_train_data('../data/OLIDv1.0')
X_train_a = vectorizer.fit_transform(clean_tweets(x_a))

clf = SVC(gamma='auto', random_state=1, kernel='linear')
# clf = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=0)
clf.fit(X_train_a, y_a)

X_test_a, y_test_a = load_test_data('../data/OLIDv1.0/testset-levela.tsv',
                                    '../data/OLIDv1.0/labels-levela.csv')
X_test_a = vectorizer.transform(clean_tweets(X_test_a))
pred = clf.predict(X_test_a)
print(classification_report(y_test_a, pred))
top = 0
bt = 0
for i in range(len(pred)):
    bt += 1
    if pred[i] == y_test_a[i]:
        top += 1
print(float(top / bt))

# Task B
new_y_b = []
new_x_b = []
for i in range(len(x_a)):
    if y_b[i] == 'TIN' or y_b[i] == 'UNT':
        new_y_b.append(y_b[i])
        new_x_b.append(x_a[i])

X_train_b = vectorizer.fit_transform(clean_tweets(new_x_b))
clf = SVC(gamma='auto', random_state=1, kernel='linear')
# clf = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=0)
clf.fit(X_train_b, new_y_b)
X_test_b, y_test_b = load_test_data('../data/OLIDv1.0/testset-levelb.tsv',
                                    '../data/OLIDv1.0/labels-levelb.csv')

X_test_b = vectorizer.transform(clean_tweets(X_test_b))
pred = clf.predict(X_test_b)
print(classification_report(y_test_b, pred))
for i in range(len(pred)):
    bt += 1
    if pred[i] == y_test_b[i]:
        top += 1
print(float(top / bt))