import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns # Karışıklık matrisi (Confusion Matrix) çizimi için eklendi
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import tensorflow as tf

# EKRAN KARTININ ÇÖKMESİNİ ENGELLEYEN AYAR
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # Hafızayı bir anda değil, ihtiyaç duydukça azar azar kullan
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- AYARLAR ---
DATA_PATH = os.path.join('TID_Verileri') 
actions = np.array([
    'A', 'B', 'C', 'C_alt', 'D', 'E', 'F', 'G', 'G_yumusak', 'H', 
    'I_noktasiz', 'I_noktali', 'J', 'K', 'L', 'M', 'N', 'O', 'O_noktali', 'P', 
    'R', 'S', 'S_alt', 'T', 'U', 'U_noktali', 'V', 'Y', 'Z', 'Bos'
])
no_sequences = 200 # Artık 200 videomuz var (100 sen + 100 annen)
sequence_length = 30 
# ---------------

print("1. Veriler Diskten Okunuyor... Lutfen Bekle (Bu islem biraz surebilir)")

label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# --- VERİYİ BÖLME (TRAIN, TEST, VALIDATION) ---
# %10'unu test için (modelin hiç görmeyeceği veriler) ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

print("\n2. Yapay Zeka (LSTM) Beyni Insa Ediliyor...")
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,126)))
# Dropout eklendi: Modelin ezberlemesini (Overfitting) zorlaştırmak için nöronların %20'si rastgele kapatılır
model.add(Dropout(0.2)) 
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# --- OPTİMİZASYON PARAMETRELERİ ---
# Öğrenme hızı (Learning Rate) manuel olarak 0.001'e ayarlandı.
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# --- EARLY STOPPING & MODEL CHECKPOINT ---
# Patience=15: Model 15 tur boyunca gelişemezse eğitimi otomatik keser.
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
# Sadece en yüksek doğruluk oranına sahip olan anı kaydeder.
checkpoint = ModelCheckpoint('en_iyi_tid_model.h5', monitor='val_categorical_accuracy', save_best_only=True, mode='max', verbose=1)

print("\n3. EGITIM BASLIYOR! (Arkana yaslan ve sayilari izle)")
# validation_split=0.1: Kalan verinin %10'u doğrulama (val_loss) için kullanılır
history = model.fit(X_train, y_train, epochs=200, validation_split=0.1, callbacks=[early_stop, checkpoint])

print("\n4. Egitim Bitti! En iyi model 'en_iyi_tid_model.h5' olarak kaydedildi.")



# ... (Yukarıdaki eğitim kısımları aynı kalacak) ...

print("\n3. EGITIM BASLIYOR! (Arkana yaslan ve sayilari izle)")
history = model.fit(X_train, y_train, epochs=200, validation_split=0.1, callbacks=[early_stop, checkpoint])

print("\n4. Egitim Bitti! En iyi model 'en_iyi_tid_model.h5' olarak kaydedildi.")

# =====================================================================
# BURADAN SONRASINI YENİ KODLA DEĞİŞTİRDİK (ANALİZ VE KAYIT BÖLÜMÜ)
# =====================================================================

print("\n" + "="*50)
print("--- 📊 RAPOR İÇİN VERİ SETİ İSTATİSTİKLERİ ---")
print(f"Toplam Sınıf (Harf) Sayısı: {len(actions)}")
print(f"Eğitim İçin Ayrılan Veri Sayısı (X_train): {X_train.shape[0]} sekans")
print(f"Test İçin Ayrılan Veri Sayısı (X_test): {X_test.shape[0]} sekans")
print("="*50 + "\n")

# Modelin tahminlerini alalım
print("--- 🧠 TEST VERİLERİ DEĞERLENDİRİLİYOR ---")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# 1. CLASSIFICATION REPORT (Precision, Recall, F1-Score)
print("\n--- 📈 AKADEMİK BAŞARI RAPORU (F1, Precision, Recall) ---")
rapor = classification_report(y_true_classes, y_pred_classes, target_names=actions)
print(rapor)

# Raporu TXT'ye kaydet
with open('model_raporu.txt', 'w', encoding='utf-8') as f:
    f.write("--- RAPOR İÇİN VERİ SETİ İSTATİSTİKLERİ ---\n")
    f.write(f"Eğitim İçin Kullanılan Veri (X_train): {X_train.shape[0]}\n")
    f.write(f"Test İçin Saklanan Veri (X_test): {X_test.shape[0]}\n\n")
    f.write("--- AKADEMIK BASARI (F1 SCORE, PRECISION, RECALL) ---\n")
    f.write(rapor)
print("✅ Metin Raporu klasörüne 'model_raporu.txt' adıyla kaydedildi!")

# 2. EĞİTİM VE VALIDASYON GRAFİKLERİ (Öğrenme Eğrileri)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['categorical_accuracy'], label='Eğitim Başarısı (Train Acc)')
plt.plot(history.history['val_categorical_accuracy'], label='Doğrulama Başarısı (Val Acc)')
plt.title('Öğrenme Eğrisi (Accuracy)')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Hatası (Train Loss)')
plt.plot(history.history['val_loss'], label='Doğrulama Hatası (Val Loss)')
plt.title('Hata Eğrisi (Loss)')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()

# Grafiği HD kaydet ve göster
plt.savefig('ogrenme_egrileri.png', dpi=300, bbox_inches='tight')
print("✅ Öğrenme Eğrisi resmi klasöre kaydedildi!")
plt.show()

# 3. CONFUSION MATRIX (Karışıklık Matrisi) Isı Haritası
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(15, 12))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=actions, yticklabels=actions)
plt.title('Confusion Matrix (Hangi harf hangisiyle karıştırıldı?)')
plt.xlabel('Modelin Tahmin Ettiği Harf')
plt.ylabel('Gerçek Harf')

# Matrisi HD kaydet ve göster
plt.savefig('confusion_matrix_tablosu.png', dpi=300, bbox_inches='tight')
print("✅ Confusion Matrix resmi klasöre kaydedildi!")
plt.show()