import numpy as np
import cv2
import mediapipe as mp
import os

# --- AYARLAR ---
# Oynatılacak harfin klasörü
HARF = 'C'
DATA_PATH = os.path.join('TID_Verileri', HARF)

WIDTH, HEIGHT = 640, 480
LINE_COLOR = (121, 22, 76) 
POINT_COLOR = (121, 44, 250) 

mp_hands = mp.solutions.hands
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

print(f"{HARF} harfinin verileri animasyon olarak oynatılıyor...")
print("Durdurmak istersen oynatıcı penceresindeyken 'Q' tuşuna basabilirsin.")

# 0'dan 100'e kadar tüm videoları (klasörleri) gez
for sequence in range(100):
    klasor_yolu = os.path.join(DATA_PATH, str(sequence))
    
    # Eğer o numarayla klasör yoksa atla
    if not os.path.exists(klasor_yolu):
        continue

    # Her videonun içindeki 30 kareyi (0'dan 29'a) sırayla oku
    for frame_num in range(30):
        npy_path = os.path.join(klasor_yolu, f"{frame_num}.npy")
        if not os.path.exists(npy_path):
            continue

        data = np.load(npy_path)
        
        # Beyaz ekran oluştur
        image = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

        lh_data = data[:63].reshape(21, 3)
        rh_data = data[63:].reshape(21, 3)

        def draw_single_hand(hand_landmarks):
            if np.all(hand_landmarks == 0): return
            
            points = []
            for landmark in hand_landmarks:
                x_px = int(landmark[0] * WIDTH)
                y_px = int(landmark[1] * HEIGHT)
                points.append((x_px, y_px))

            for connection in HAND_CONNECTIONS:
                cv2.line(image, points[connection[0]], points[connection[1]], LINE_COLOR, 3)

            for point in points:
                cv2.circle(image, point, 5, POINT_COLOR, -1)

        draw_single_hand(lh_data)
        draw_single_hand(rh_data)

        # Ekrana hangi videoda olduğumuzu yazdıralım ki hoca görsün
        cv2.putText(image, f"Oynatilan Harf: {HARF}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(image, f"Video Sirasi: {sequence} / 100", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow('Veriseti Animasyon Oynatici', image)
        
        # 30 milisaniye bekle (videonun akış hızı). Çıkmak için Q'ya bas.
        if cv2.waitKey(30) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
print("Oynatma tamamlandı!")