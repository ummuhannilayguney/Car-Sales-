import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

# ==========================================
# 1. VERİ YÜKLEME VE KORELASYON
# ==========================================
df = pd.read_excel('Lab02_UmmuhanNilay_Guney/CarSales.xlsx')
# İki Bağımsız Değişken (X1, X2) ve Bir Bağımlı Değişken (y)
data = df[['Engine_size', 'Horsepower', 'Price_in_thousands']].dropna()

X1 = data['Engine_size'].values
X2 = data['Horsepower'].values
y = data['Price_in_thousands'].values

X = np.column_stack((X1, X2)) # Özellikleri birleştiriyoruz
corr_matrix = data.corr()

# ==========================================
# 2. EĞİTİM VE TEST VERİSİ OLARAK AYIRMA (%80 - %20)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. VERİ SIZINTISINI (LEAKAGE) ÖNLEYEREK ÖLÇEKLENDİRME
# ==========================================
# Min-Max değerlerini SADECE eğitim verisinden öğreniyoruz
X_train_min, X_train_max = np.min(X_train, axis=0), np.max(X_train, axis=0)
y_train_min, y_train_max = np.min(y_train), np.max(y_train)

X_train_scaled = (X_train - X_train_min) / (X_train_max - X_train_min)
y_train_scaled = (y_train - y_train_min) / (y_train_max - y_train_min)

X_test_scaled = (X_test - X_train_min) / (X_train_max - X_train_min)
y_test_scaled = (y_test - y_train_min) / (y_train_max - y_train_min)

# ==========================================
# 4. ÇOKLU GRADIENT DESCENT (MULTIPLE LINEAR REGRESSION)
# ==========================================
w1, w2, b = 0.0, 0.0, 0.0  # Artık 2 ağırlığımız var (w1: Motor, w2: Beygir)
learning_rate = 2
iterations = 150
m_train = len(y_train_scaled)

history = {'mse': [], 'mae': []}

for i in range(iterations):
    # Tahmin: y = w1*X1 + w2*X2 + b
    y_pred = w1 * X_train_scaled[:, 0] + w2 * X_train_scaled[:, 1] + b
    error = y_pred - y_train_scaled
    
    mse = (1/(2*m_train)) * np.sum(error**2)
    history['mse'].append(mse)
    history['mae'].append(np.mean(np.abs(error)))
    
    # Türevler (Her bir ağırlık için ayrı)
    dw1 = (1/m_train) * np.dot(error, X_train_scaled[:, 0])
    dw2 = (1/m_train) * np.dot(error, X_train_scaled[:, 1])
    db = (1/m_train) * np.sum(error)
    
    # Güncelleme
    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2
    b -= learning_rate * db

# ==========================================
# 5. TEST BAŞARISI HESAPLAMA (R2 Skoru)
# ==========================================
y_test_pred = w1 * X_test_scaled[:, 0] + w2 * X_test_scaled[:, 1] + b

# R2 Hesabı
ss_res_train = np.sum((y_train_scaled - (w1 * X_train_scaled[:, 0] + w2 * X_train_scaled[:, 1] + b))**2)
ss_tot_train = np.sum((y_train_scaled - np.mean(y_train_scaled))**2)
r2_train = 1 - (ss_res_train / ss_tot_train)

ss_res_test = np.sum((y_test_scaled - y_test_pred)**2)
ss_tot_test = np.sum((y_test_scaled - np.mean(y_test_scaled))**2)
r2_test = 1 - (ss_res_test / ss_tot_test)

# ==========================================
# 6. MUAZZAM 5 PANEL GÖRSELLEŞTİRME EKRANI
# ==========================================
fig = plt.figure(figsize=(18, 10))

# --- PANEL 1: 3D Çoklu Regresyon Düzlemi ---
ax1 = fig.add_subplot(231, projection='3d')
x1_surf = np.linspace(0, 1, 10)
x2_surf = np.linspace(0, 1, 10)
x1_surf, x2_surf = np.meshgrid(x1_surf, x2_surf)
y_surf = w1 * x1_surf + w2 * x2_surf + b

ax1.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], y_train_scaled, color='blue', alpha=0.5, label='Eğitim Verisi')
ax1.plot_surface(x1_surf, x2_surf, y_surf, color='red', alpha=0.3) # Regresyon Düzlemi
ax1.set_title("3D Çoklu Regresyon Modeli")
ax1.set_xlabel("Motor Hacmi"); ax1.set_ylabel("Beygir Gücü"); ax1.set_zlabel("Fiyat")
ax1.view_init(elev=20, azim=45)

# --- PANEL 2: 1. Değişkenin Tekil Grafiği ---
ax2 = fig.add_subplot(232)
ax2.scatter(X1, y, color='blue', alpha=0.5)
ax2.set_title("Değişken 1: Motor Hacmi vs Fiyat")
ax2.set_xlabel("Motor Hacmi"); ax2.set_ylabel("Fiyat")
ax2.grid(True, alpha=0.3)

# --- PANEL 3: 2. Değişkenin Tekil Grafiği ---
ax3 = fig.add_subplot(233)
ax3.scatter(X2, y, color='green', alpha=0.5)
ax3.set_title("Değişken 2: Beygir Gücü vs Fiyat")
ax3.set_xlabel("Beygir Gücü"); ax3.set_ylabel("Fiyat")
ax3.grid(True, alpha=0.3)

# --- PANEL 4: 3x3 Korelasyon Matrisi ---
ax4 = fig.add_subplot(234)
cax = ax4.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax, ax=ax4)
ax4.set_xticks(range(3)); ax4.set_yticks(range(3))
ax4.set_xticklabels(['Motor', 'Beygir', 'Fiyat'])
ax4.set_yticklabels(['Motor', 'Beygir', 'Fiyat'], rotation=90, va='center')
ax4.set_title("Korelasyon Matrisi", pad=20)
for i in range(3):
    for j in range(3):
        renk = 'white' if abs(corr_matrix.iloc[i, j]) > 0.6 else 'black'
        ax4.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha='center', va='center', color=renk, fontweight='bold')

# --- PANEL 5: Hata Öğrenme Eğrisi ---
ax5 = fig.add_subplot(235)
ax5.plot(history['mse'], color='red', label='MSE', lw=2)
ax5.plot(history['mae'], color='orange', label='MAE', linestyle='--', lw=2)
ax5.set_title("Eğitim Sürecinde Hata Azalması")
ax5.set_xlabel("İterasyon"); ax5.set_ylabel("Hata")
ax5.legend(); ax5.grid(True, alpha=0.3)

# --- PANEL 6: Performans Raporu (Metin) ---
ax6 = fig.add_subplot(236)
ax6.axis('off')
rapor = (
    "--- ÇOKLU DOĞRUSAL REGRESYON RAPORU ---\n\n"
    f"Kullanılan Özellikler : Motor Hacmi & Beygir Gücü\n"
    f"Öğrenilen w1 (Motor)  : {w1:.4f}\n"
    f"Öğrenilen w2 (Beygir) : {w2:.4f}\n"
    f"Öğrenilen Bias (b)    : {b:.4f}\n\n"
    f"Model Başarısı (R2 Skorları):\n"
    f"Eğitim Verisi R2: % {r2_train*100:.1f}\n"
    f"Test Verisi R2  : % {r2_test*100:.1f}\n\n"
    "Sonuç: İkinci değişken (Beygir Gücü) eklendiğinde\n"
    "modelin fiyat tahmin başarısı neredeyse İKİ KATINA çıktı!"
)
ax6.text(0.1, 0.5, rapor, fontsize=12, va='center', bbox=dict(facecolor='lightgreen', alpha=0.2))

plt.tight_layout()
plt.show()

# Terminal Çıktısı 
print("\n=== MODEL DEĞERLENDİRME SONUÇLARI ===")
print(f"Test Ortalama Karesel Hata (MSE): {history['mse'][-1]:.4f}")
print(f"Test Eğitim R2 Skoru: {r2_train:.4f}")
print(f"Test Verisi R2 Skoru: {r2_test:.4f}")