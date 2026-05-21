import cv2
import numpy as np
import os
import mediapipe as mp
import sys
import time 

# --- MEDIAPIPE AYARLARI ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- KLASÖR AYARLARI ---
DATA_PATH = os.path.join('TID_Verileri') 
REPORT_PATH = os.path.join('TID_Rapor_Resimleri') 

# =======================================================
# --- ÇEKİM AYARLARI (BURAYI KİM ÇEKİYORSA ONA GÖRE DEĞİŞTİR) ---

BASLANGIC_VIDEO = 100
BITIS_VIDEO = 200
# =======================================================

# TOPLANACAK HARFLER VE VİDEO AYARLARI
# 29 HARF + BOS SINIFI (Klasör hatalarını önlemek için İngilizce karakterli)
actions = np.array([
    'A', 'B', 'C', 'C_alt', 'D', 'E', 'F', 'G', 'G_yumusak', 'H', 
    'I_noktasiz', 'I_noktali', 'J', 'K', 'L', 'M', 'N', 'O', 'O_noktali', 'P', 
    'R', 'S', 'S_alt', 'T', 'U', 'U_noktali', 'V', 'Y', 'Z', 'Bos'
])
sequence_length = 30  

os.makedirs(REPORT_PATH, exist_ok=True)

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4) as holistic:
    for action in actions:
        dir_path = os.path.join(DATA_PATH, action)
        os.makedirs(dir_path, exist_ok=True)
        
        mevcut_videolar = [int(f) for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f)) and f.isdigit()]
        baslangic_videosu = max(mevcut_videolar) + 1 if mevcut_videolar else 0
        
        # Sistemin güvenliği: Kaldığı yerden devam etmesi veya o bloğu atlaması için
        gercek_baslangic = max(baslangic_videosu, BASLANGIC_VIDEO)

        if gercek_baslangic >= BITIS_VIDEO:
            print(f"--- {action} harfi bu blok icin ({BASLANGIC_VIDEO}-{BITIS_VIDEO}) zaten tamamlanmis. Atlaniyor... ---")
            continue

        report_image_captured = False

        # 1. HARFE BAŞLAMADAN ÖNCEKİ İLK BEKLEME EKRANI ('N' tuşu)
       # 1. HARFE BAŞLAMADAN ÖNCEKİ İLK BEKLEME EKRANI ('N' tuşu)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1) 
            
            # --- RADAR EKLENTİSİ ---
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            # -----------------------
            
            cv2.putText(image, f"SIRADAKI HARF: {action}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
            cv2.putText(image, f"(Cekilecek Blok: {BASLANGIC_VIDEO} - {BITIS_VIDEO})", (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            cv2.putText(image, "Hazirsan BASLAMAK icin 'N' tusuna bas", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
            
            cv2.imshow('TID Veri Toplama', image)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('n') or key == ord('N'): 
                break
            elif key == ord('q') or key == ord('Q'): 
                cap.release()
                cv2.destroyAllWindows()
                sys.exit()

        # 3 SANİYE HAZIRLIK
        baslangic_zamani = time.time()
        while time.time() - baslangic_zamani < 3:
            ret, frame = cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            kalan_sure = int(3 - (time.time() - baslangic_zamani)) + 1
            cv2.putText(frame, f"HAZIRLAN... {kalan_sure}", (400, 350), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            cv2.imshow('TID Veri Toplama', frame)
            cv2.waitKey(10)

        # KAYIT DÖNGÜSÜ
      # KAYIT DÖNGÜSÜ
        for sequence in range(gercek_baslangic, BITIS_VIDEO):
            
            # --- MANUEL MOLA SİSTEMİ ---
            if (sequence - gercek_baslangic) > 0 and (sequence - gercek_baslangic) % 20 == 0:
                # D tuşuna basana kadar sonsuz bekleme
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: continue
                    frame = cv2.flip(frame, 1)
                    
                    # --- MOLA  EKLENTİSİ ---
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    if results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    if results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    
                    cv2.putText(image, "MOLA! POZISYONUNU DEGISTIR", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    cv2.putText(image, "Hazir olunca DEVAM etmek icin 'D' tusuna bas", (100, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow('TID Veri Toplama', image)
                    
                    # CLAUDE FIX 1: waitKey sadece 1 kere çağrılır!
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('d') or key == ord('D'):
                        break
                        
                # D'ye bastıktan sonra 3 saniye geri sayım
                mola_zamani = time.time()
                while time.time() - mola_zamani < 3:
                    ret, frame = cap.read()
                    if not ret: continue
                    frame = cv2.flip(frame, 1)
                    kalan_mola = int(3 - (time.time() - mola_zamani)) + 1
                    cv2.putText(frame, f"KAYIT BASLIYOR... {kalan_mola}", (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                    cv2.imshow('TID Veri Toplama', frame)
                    cv2.waitKey(10)
            # ----------------------------------------

            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)
            frame_num = 0
            
            sequence_data = [] 
            kaydedilecek_resim = None
            
            # 30 KAREYİ KAYDETME DÖNGÜSÜ
            while frame_num < sequence_length:
                ret, frame = cap.read()
                if not ret: continue
                
                frame = cv2.flip(frame, 1)

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                el_gorunuyor_mu = results.left_hand_landmarks or results.right_hand_landmarks

                if el_gorunuyor_mu:
                    if results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    if results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                else:
                    # El görünmese de ekrana uyarı yaz ama KAYDA DEVAM ET
                    cv2.putText(image, 'EL BULUNAMADI!', (50,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

                cv2.putText(image, f'KAYITTA... {action} | Video: {sequence}/{BITIS_VIDEO}', (15,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                
                # CLAUDE FIX 2: El görünse de görünmese de veriyi listeye at ve frame'i artır! (Zaman bükülmesini önler)
                keypoints = extract_keypoints(results)
                sequence_data.append(keypoints)
                frame_num += 1
                
                if not report_image_captured and (sequence - gercek_baslangic) >= 2 and frame_num == 2:
                    kaydedilecek_resim = image.copy()
                    report_image_captured = True

                cv2.imshow('TID Veri Toplama', image)
                
                # Buradaki waitKey için de aynı fix
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit()

            # 30 kare bittikten sonra RAM'den Diske Yaz!
            for i, data in enumerate(sequence_data):
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(i))
                np.save(npy_path, data)
                
            if kaydedilecek_resim is not None:
                resim_adi = os.path.join(REPORT_PATH, f'Hoca_Rapor_{action}.jpg')
                if not os.path.exists(resim_adi):
                    cv2.imwrite(resim_adi, kaydedilecek_resim)
cap.release()
cv2.destroyAllWindows()
