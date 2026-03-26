import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# 1. VERİYİ YÜKLEME VE HAZIRLIK
# Dosya adını ortamına göre 'CarSales.xlsx' veya '.csv' olarak değiştirebilirsin.
df = pd.read_csv('CarSales.xlsx - Car_sales.csv.csv')

# Kullanılacak sütunlar
features = ['Horsepower', 'Engine_size', 'Curb_weight', 'Price_in_thousands']

# İçinde boş/eksik (NaN) değer olan satırları temizliyoruz
df_clean = df[features].dropna()

# Bağımsız değişkenler (X) ve Bağımlı Değişken (y - Fiyat)
X = df_clean[['Horsepower', 'Engine_size', 'Curb_weight']]
y = df_clean['Price_in_thousands']

# Modeli test edebilmek için veriyi %80 Eğitim, %20 Test olarak ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sonuçları tutacağımız liste
sonuclar = []

# ==========================================
# A. BASİT DOĞRUSAL REGRESYON (Simple Linear)
# Sadece Horsepower kullanılarak
# ==========================================
X_train_simple = X_train[['Horsepower']]
X_test_simple = X_test[['Horsepower']]

model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train)
y_pred_simple = model_simple.predict(X_test_simple)

mse_simple = mean_squared_error(y_test, y_pred_simple)
r2_simple = r2_score(y_test, y_pred_simple)
sonuclar.append(('Simple Linear', mse_simple, r2_simple))

# Grafik A
plt.figure(figsize=(8, 5))
plt.scatter(X_test_simple, y_test, color='blue', label='Gerçek Veriler', alpha=0.6)
plt.plot(X_test_simple, y_pred_simple, color='red', label='Regresyon Doğrusu (Tahmin)')
plt.title('A. Basit Doğrusal Regresyon: Beygir Gücü vs Fiyat')
plt.xlabel('Horsepower (Beygir Gücü)')
plt.ylabel('Price in thousands (Fiyat)')
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# B. ÇOKLU DOĞRUSAL REGRESYON (Multiple Linear)
# Horsepower, Engine_size, Curb_weight kullanılarak
# ==========================================
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)
y_pred_multi = model_multi.predict(X_test)

mse_multi = mean_squared_error(y_test, y_pred_multi)
r2_multi = r2_score(y_test, y_pred_multi)
sonuclar.append(('Multiple Linear', mse_multi, r2_multi))
# Not: 3 boyutlu bir veri olduğu için standart 2D grafiği çizilmez, atlanmıştır.

# ==========================================
# C. POLİNOMİYAL REGRESYON (Degree: 3)
# Görselleştirmeyi net görebilmek için Horsepower üzerinden yapıyoruz
# ==========================================
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train_simple)
X_test_poly = poly.transform(X_test_simple)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)
y_pred_poly = model_poly.predict(X_test_poly)

mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)
sonuclar.append(('Polynomial (Deg:3)', mse_poly, r2_poly))

# Grafik C (Eğrinin düzgün çizilebilmesi için verileri sıralıyoruz)
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X_test_simple.values, y_pred_poly), key=sort_axis)
X_test_sorted, y_poly_pred_sorted = zip(*sorted_zip)

plt.figure(figsize=(8, 5))
plt.scatter(X_test_simple, y_test, color='blue', label='Gerçek Veriler', alpha=0.6)
plt.plot(X_test_sorted, y_poly_pred_sorted, color='green', linewidth=2, label='Polinom Eğrisi (Tahmin)')
plt.title('C. Polinomiyal Regresyon (Derece=3): Beygir Gücü vs Fiyat')
plt.xlabel('Horsepower (Beygir Gücü)')
plt.ylabel('Price in thousands (Fiyat)')
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# D. RIDGE REGRESYON
# Aşırı öğrenmeyi engellemek için alpha=1.0 parametresi test ediliyor
# ==========================================
model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_train, y_train)
y_pred_ridge = model_ridge.predict(X_test)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
sonuclar.append(('Ridge Regression', mse_ridge, r2_ridge))

# ==========================================
# 3. SONUÇLARIN KIYASLANMASI TABLOSU
# ==========================================
print("\n--- 3. SONUÇLARIN KIYASLANMASI ---")
sonuclar_df = pd.DataFrame(sonuclar, columns=['Model Türü', 'MSE (Hata)', 'R-Squared'])
print(sonuclar_df.to_string(index=False))


# https://github.com/ummuhannilayguney