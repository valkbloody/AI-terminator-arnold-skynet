import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv("diamonds.csv", sep=",")
df.drop('Unnamed: 0', axis=1, inplace=True)
order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
df['cut'] = pd.Categorical(df["cut"], categories=order, ordered=True).codes + 1
clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
color_order = ["J", "I", "H", "G", "F", "E", "D"]
df['color'] = pd.Categorical(df["color"], categories=color_order, ordered=True).codes + 1
df['clarity'] = pd.Categorical(df["clarity"], categories=clarity_order, ordered=True).codes + 1

X = df.drop("price", axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

vif_data = pd.DataFrame()
vif_data["Переменная"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

vif_data = vif_data.sort_values(by="VIF", ascending=False)

print(vif_data)