import numpy as np

# Herhangi bir kaydettiğin .npy dosyasının yolunu buraya yaz
# Bilgisayarındaki yola göre klasör isimlerini doğru yazdığından emin ol
koordinatlar = np.load('TID_Verileri/A/0/0.npy')

print("İşte el iskeletinin X, Y, Z koordinat matrisi:")
print(koordinatlar)
print(f"\nToplam {len(koordinatlar)} adet değer var.")