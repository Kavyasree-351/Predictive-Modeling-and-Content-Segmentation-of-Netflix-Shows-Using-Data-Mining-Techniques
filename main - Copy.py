import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("movies_clean.csv")

id_cols = []
for col in ["show_id", "title"]:
    if col in df.columns:
        id_cols.append(col)

if "revenue" not in df.columns:
    raise ValueError("Column 'revenue' not found in movies_clean.csv – required for ROI-based labels.")

df = df[(df["budget"] > 0) & (df["revenue"] > 0)]

le_genre = LabelEncoder()
le_country = LabelEncoder()
le_lang = LabelEncoder()

df["primary_genre_enc"] = le_genre.fit_transform(df["primary_genre"])
df["primary_country_enc"] = le_country.fit_transform(df["primary_country_flagged"])
df["language_enc"] = le_lang.fit_transform(df["language_flagged"])

df["ROI"] = df["revenue"] / df["budget"]

def label_movie_by_roi(roi):
    if roi >= 2:
        return "Hit"
    elif roi >= 1:
        return "Average"
    else:
        return "Flop"

df["success_class"] = df["ROI"].apply(label_movie_by_roi)

class_map = {"Flop": 0, "Average": 1, "Hit": 2}
df["success_class"] = df["success_class"].map(class_map)

features = [
    "budget", "popularity", "vote_average", "vote_count",
    "primary_genre_enc", "primary_country_enc", "language_enc"
]

df = df.dropna(subset=features + ["success_class"])

X = df[features]
y = df["success_class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

test_indices = X_test.index

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "DT": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "NB": GaussianNB(),
    "SVM": SVC(kernel="rbf")
}

predictions = {}

print("\n===== RESULTS (ROI-based success: Flop / Average / Hit) =====\n")

for name, model in models.items():
    print(f"\n=== {name} MODEL ===")
    if name in ["KNN", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    predictions[name] = y_pred
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Flop", "Average", "Hit"]))

def custom_predict(row):
    if row["vote_average"] >= 7.5 and row["popularity"] >= 50:
        return 2
    if row["vote_average"] < 5:
        return 0
    return 1

pred_CUSTOM = X_test.apply(custom_predict, axis=1).values

print("\n=== CUSTOM HEURISTIC MODEL ===")
print(f"Accuracy: {accuracy_score(y_test, pred_CUSTOM):.4f}")
print(classification_report(y_test, pred_CUSTOM, target_names=["Flop", "Average", "Hit"]))

cols_for_output = id_cols + [
    "budget", "popularity", "vote_average", "vote_count"
]

output = df.loc[test_indices, cols_for_output].copy()

output["primary_genre"] = df.loc[test_indices, "primary_genre"]
output["primary_country_flagged"] = df.loc[test_indices, "primary_country_flagged"]
output["language_flagged"] = df.loc[test_indices, "language_flagged"]

output["ROI"] = df.loc[test_indices, "ROI"]

output["true_success_class"] = y_test.values
output["pred_DT"] = predictions["DT"]
output["pred_KNN"] = predictions["KNN"]
output["pred_NB"] = predictions["NB"]
output["pred_SVM"] = predictions["SVM"]
output["pred_CUSTOM"] = pred_CUSTOM

class_map_reverse = {0: "Flop", 1: "Average", 2: "Hit"}

output["true_success_class"] = output["true_success_class"].map(class_map_reverse)
output["pred_DT"] = output["pred_DT"].map(class_map_reverse)
output["pred_KNN"] = output["pred_KNN"].map(class_map_reverse)
output["pred_NB"] = output["pred_NB"].map(class_map_reverse)
output["pred_SVM"] = output["pred_SVM"].map(class_map_reverse)
output["pred_CUSTOM"] = output["pred_CUSTOM"].map(class_map_reverse)

output.to_csv("movie_test_predictions.csv", index=False)
print("\nOutput saved as movie_test_predictions.csv")
