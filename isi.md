# 🎯 SCikit-learn MASTER PREHĽAD

---

## 🧠 1. Generovanie a delenie dát

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split

# generovanie dát
X, y = datasets.make_classification(
    n_samples=1000,
    n_features=3,
    n_redundant=0
)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=0
)
```

---

## ⚙️ 2. Predspracovanie dát (Preprocessing)

### 2.1 Normalizácia a štandardizácia
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

StandardScaler()                       # štandardizácia (z-score)
MinMaxScaler(feature_range=(0, 1))     # škálovanie do rozsahu 0–1
Normalizer(norm='l2')                  # úprava smerového typu (L1/L2)
```

#### Použitie
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit+transform na TRAIN
X_test_scaled = scaler.transform(X_test)         # len transform na TEST
```
🧠 `fit_transform()` → iba na X_train  
🧠 `transform()` → iba na X_test  
🧠 nikdy nepoužívame na y

---

### 2.2 Kvantilové a power transformácie
```python
from sklearn.preprocessing import QuantileTransformer, PowerTransformer

QuantileTransformer(method="uniform", standardize=True)
PowerTransformer(method="yeo-johnson")
```

---

### 2.3 Kódovanie kategórií
```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

OneHotEncoder(handle_unknown='ignore', sparse_output=False)
LabelEncoder()  # používa sa len na cieľ y
```

---

### 2.4 Imputácia chýbajúcich hodnôt
```python
from sklearn.impute import SimpleImputer

SimpleImputer(strategy='median')
```

---

## 🌳 3. Modely (Algoritmy učenia)

### Rozhodovacie stromy
```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

DecisionTreeClassifier(criterion="gini", max_depth=None, random_state=42)
DecisionTreeRegressor(criterion="squared_error", max_depth=None, random_state=42)
```

### Náhodný les
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

RandomForestClassifier(n_estimators=200, random_state=42)
RandomForestClassifier(n_estimators=100, random_state=100, max_features='sqrt')
RandomForestRegressor(n_estimators=100, random_state=100)
```

### K‑Najbližších susedov
```python
from sklearn.neighbors import KNeighborsClassifier

KNeighborsClassifier(metric='cosine', n_neighbors=5)
```

### Support Vector Machines (SVM)
```python
from sklearn.svm import SVC, SVR

SVC(kernel='linear', C=1.0, random_state=42)
SVR(kernel='linear', C=1.0)
```

### Logistická regresia
```python
from sklearn.linear_model import LogisticRegression

LogisticRegression(max_iter=1000, random_state=42)
```

---

## 📈 4. Výber vlastností (Feature selection)
```python
from sklearn.feature_selection import SelectKBest, f_regression

SelectKBest(score_func=f_regression, k=10)
```

---

## 🔁 5. Validácia a krížová validácia
```python
from sklearn.model_selection import KFold, RepeatedStratifiedKFold

KFold(n_splits=5)
RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
```

🧠 n_splits — počet foldov  
🧠 n_repeats — počet opakovaní rozdelení

---

## 🔍 6. Vyhľadávanie parametrov (Grid Search)
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", 0.01, 0.001]
}

grid = GridSearchCV(
    svc, param_grid=param_grid, scoring='accuracy', cv=5
)

grid.best_estimator_
grid.best_params_
grid.best_score_
```

---

## 🎯 7. Vyhodnocovanie modelu (Metriky a reporty)

### Klasifikačné metriky
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
rep = classification_report(y_test, y_pred)
```

### Regresné metriky
```python
import sklearn.metrics as metrics

metrics.r2_score(y_test, y_pred)
metrics.mean_absolute_error(y_test, y_pred)
```

---

## 🔗 8. Pipeline – fitovanie a predikcia naraz
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression())
])

pipe.fit(X_train, y_train)      # Fitne scaler aj model
y_pred = pipe.predict(X_test)   # Automaticky transformuje test a predikuje
```

🧠 V `Pipeline`:
- `fit()` → fitne všetky kroky + model
- `predict()` → automaticky transformuje a predikuje

---

## 🧮 9. Python utility
```python
import heapq

heapq.heappush(heap, (priority, counter, data))
```

---

## ✅ ZHRNUTIE HLAVNÝCH MYŠLIENOK

| Operácia | Na čo sa používa | Voláš na |
|-----------|------------------|----------|
| fit_transform() | naučí + transformuje dáta | X_train |
| transform() | použije rovnaké nastavenia | X_test |
| fit() | naučí model | X_train, y_train |
| predict() | predpovede | X_test |
| Pipeline.fit() | fitne všetky kroky | X_train, y_train |
| Pipeline.predict() | transformuje + predikuje | X_test |