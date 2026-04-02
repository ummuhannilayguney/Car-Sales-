import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ==========================================
# 1. VERİ YÜKLEME VE KORELASYON
# ==========================================
df = pd.read_excel('Lab02/CarSales.xlsx')
data = df[['Engine_size', 'Price_in_thousands']].dropna()

# Bağımsız değişken (X: Motor Hacmi), Bağımlı değişken (y: Fiyat)
X = data['Engine_size'].values
y = data['Price_in_thousands'].values

# Değişkenler arası ilişki matrisini alalım
corr_matrix = data.corr()

# ==========================================
# 2. TRAIN VE TEST OLARAK AYIRMA (%80 Eğitim, %20 Test)
# ==========================================
# random_state=42: Her çalıştırdığımızda aynı ayrımı yapsın diye eklendi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. VERİ ÖLÇEKLENDİRME (Data Leakage'i Önleyerek)
# ==========================================
# KURAL: Min ve Max değerleri SADECE Eğitim (Train) setinden öğrenilir!
X_min, X_max = np.min(X_train), np.max(X_train)
y_min, y_max = np.min(y_train), np.max(y_train)

X_train_scaled = (X_train - X_min) / (X_max - X_min)
y_train_scaled = (y_train - y_min) / (y_max - y_min)

X_test_scaled = (X_test - X_min) / (X_max - X_min)
y_test_scaled = (y_test - y_min) / (y_max - y_min)

# ==========================================
# 4. GRADIENT DESCENT DÖNGÜSÜ (SADECE EĞİTİM VERİSİ İLE)
# ==========================================
w, b = 0.0, 0.0      
learning_rate = 0.001 
iterations = 500     
m_train = len(y_train_scaled)

history = {'w': [], 'b': [], 'mse': [], 'mae': []}

for i in range(iterations):
    y_pred = w * X_train_scaled + b
    error = y_pred - y_train_scaled
    
    # Eğitim hata metrikleri
    mse = (1/(2*m_train)) * np.sum(error**2)
    mae = np.mean(np.abs(error)) 
    
    # Türev alma ve parametreleri güncelleme
    dw = (1/m_train) * np.dot(error, X_train_scaled)
    db = (1/m_train) * np.sum(error)
    
    w -= learning_rate * dw
    b -= learning_rate * db
    
    history['w'].append(w); history['b'].append(b)
    history['mse'].append(mse); history['mae'].append(mae)

# ==========================================
# 5. TEST VERİSİ ÜZERİNDE MODELİ DEĞERLENDİRME
# ==========================================
# Optimize edilmiş w ve b değerlerini kullanarak Test verisinde tahmin yapıyoruz
y_test_pred = w * X_test_scaled + b

# R2 Skorlarını hesaplama
ss_res_train = np.sum((y_train_scaled - (w * X_train_scaled + b))**2)
ss_tot_train = np.sum((y_train_scaled - np.mean(y_train_scaled))**2)
r2_train = 1 - (ss_res_train / ss_tot_train)

ss_res_test = np.sum((y_test_scaled - y_test_pred)**2)
ss_tot_test = np.sum((y_test_scaled - np.mean(y_test_scaled))**2)
r2_test = 1 - (ss_res_test / ss_tot_test)

# ==========================================
# 6. GÖRSELLEŞTİRME (4 Panelli Dashboard)
# ==========================================
fig = plt.figure(figsize=(16, 12))

# --- 1. Bölme (Sol Üst): 3D Kayıp Yüzeyi (Eğitim Verisi Üzerinde) ---
ax1 = fig.add_subplot(221, projection='3d')
W_vals = np.linspace(-0.5, 1.0, 50); B_vals = np.linspace(-0.5, 0.5, 50)
W_grid, B_grid = np.meshgrid(W_vals, B_vals)
Z_grid = np.array([np.mean(((w_g*X_train_scaled + b_g) - y_train_scaled)**2)/2 for w_g, b_g in zip(np.ravel(W_grid), np.ravel(B_grid))]).reshape(W_grid.shape)

ax1.plot_surface(W_grid, B_grid, Z_grid, cmap='viridis', alpha=0.7)
ax1.plot(history['w'], history['b'], history['mse'], color='red', marker='.', markersize=4, label='İniş Yolu')
ax1.set_title("3D Kayıp Yüzeyi (Eğitim Seti)")
ax1.set_xlabel("Ağırlık (w)"); ax1.set_ylabel("Sapma (b)"); ax1.set_zlabel("MSE")
ax1.view_init(elev=30, azim=-45)

# --- 2. Bölme (Sağ Üst): 2D Kontur ---
ax2 = fig.add_subplot(222)
ax2.contour(W_grid, B_grid, Z_grid, levels=30, cmap='viridis')
ax2.plot(history['w'], history['b'], color='red', marker='o', markersize=3, label='Doğal Gradyan İnişi')
ax2.set_title("w-b Uzayında Kontur (Kuşbakışı)")
ax2.set_xlabel("Ağırlık (w)"); ax2.set_ylabel("Sapma (b)")
ax2.grid(True, linestyle='--', alpha=0.5); ax2.legend()

# --- 3. Bölme (Sol Alt): Eğitim Hata Metrikleri ve Rapor ---
ax3 = fig.add_subplot(223)
ax3.plot(history['mse'], label='Train MSE', color='red', lw=2)
ax3.plot(history['mae'], label='Train MAE', color='blue', linestyle='--', lw=2)
ax3.set_title("Modelin Öğrenme Eğrisi")
ax3.set_xlabel("Adım (İterasyon)"); ax3.set_ylabel("Hata Oranı")
ax3.grid(True, linestyle='--', alpha=0.5); ax3.legend()

# Test Başarısını Grafiğe Ekleyelim
rapor_metni = f"BAŞARI RAPORU:\nEğitim R2 Skoru: %{r2_train*100:.1f}\nTest R2 Skoru: %{r2_test*100:.1f}"
ax3.text(40, max(history['mse'])*0.7, rapor_metni, fontsize=12, bbox=dict(facecolor='yellow', alpha=0.3))

# --- 4. Bölme (Sağ Alt): Korelasyon Matrisi ---
ax4 = fig.add_subplot(224)
cax = ax4.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax, ax=ax4)
ax4.set_xticks([0, 1]); ax4.set_yticks([0, 1])
ax4.set_xticklabels(['Motor Hacmi', 'Fiyat'], fontsize=11)
ax4.set_yticklabels(['Motor Hacmi', 'Fiyat'], fontsize=11)
ax4.set_title("Değişkenler Arası Korelasyon Matrisi (Pearson)", pad=20, fontsize=12)

for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        deger = corr_matrix.iloc[i, j]
        renk = 'white' if abs(deger) > 0.5 else 'black'
        ax4.text(j, i, f"{deger:.2f}", ha='center', va='center', color=renk, fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# Terminal Çıktısı İçin Özet
print("-" * 40)
print(f"EĞİTİM TAMAMLANDI!")
print(f"Kullanılan Eğitim Verisi: {len(X_train)} adet")
print(f"Kullanılan Test Verisi: {len(X_test)} adet")
print("-" * 40)
print(f"Bulunan Ağırlık (w): {w:.4f}")
print(f"Bulunan Sapma (b): {b:.4f}")
print(f"Eğitim (Train) Seti Başarısı (R2): %{r2_train*100:.1f}")
print(f"Test Seti Başarısı (R2): %{r2_test*100:.1f}")
print("-" * 40)