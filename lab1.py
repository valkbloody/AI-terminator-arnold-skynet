import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import seaborn as sns
import numpy as np

df = pd.read_csv("diamonds.csv", sep=",")
df.drop('Unnamed: 0', axis=1, inplace=True)
order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
df['cut'] = pd.Categorical(df["cut"], categories=order, ordered=True).codes + 1
clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
color_order = ["J", "I", "H", "G", "F", "E", "D"]
df['color'] = pd.Categorical(df["color"], categories=color_order, ordered=True).codes + 1
df['clarity'] = pd.Categorical(df["clarity"], categories=clarity_order, ordered=True).codes + 1

print("Размер датасета:", df.shape)
print("\nПервые строки:")
print(df.head())
print("\nИнформация о данных:")
print(df.info())
print("\nОписание признаков:")
print(df.describe())

print("\nКоличество пропусков:")
print(df.isnull().sum())

numeric_columns = df.select_dtypes(include=[np.number]).columns

data = {
    'Параметр': [],
    'Асимметрия': [],
    'Эксцесс': []
}

for col in numeric_columns:
    skew_val = skew(df[col].dropna())
    kurt_val = kurtosis(df[col].dropna())
    data['Параметр'].append(col)
    data['Асимметрия'].append(skew_val)
    data['Эксцесс'].append(kurt_val)

summary_df = pd.DataFrame(data)
print(summary_df)
count_of_bins = int(1 + 3.322*math.log10(df.shape[0]))

plt.figure(figsize=(6,5))
df['price'].hist(bins=100,edgecolor="black",color="pink")
plt.ylabel('Количество')
plt.xlabel('Цена')
plt.title("Цена алмаза")
plt.show()

df.hist(bins=count_of_bins, figsize=(15,10), edgecolor="black")
plt.suptitle("Распределение признаков")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Корреляционная матрица признаков")
plt.show()

print(df.corr())