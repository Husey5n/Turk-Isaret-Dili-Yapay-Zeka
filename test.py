import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf 
from tensorflow.keras.models import load_model

# --- EKRAN KARTININ ÇÖKMESİNİ ENGELLEYEN KALKAN ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# --------------------------------------------------

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) 

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

print("Yapay Zeka Beyni ('en_iyi_tid_model.h5') Yukleniyor... Lutfen Bekle...")
model = load_model('en_iyi_tid_model.h5')

actions = np.array([
    'A', 'B', 'C', 'C_alt', 'D', 'E', 'F', 'G', 'G_yumusak', 'H', 
    'I_noktasiz', 'I_noktali', 'J', 'K', 'L', 'M', 'N', 'O', 'O_noktali', 'P', 
    'R', 'S', 'S_alt', 'T', 'U', 'U_noktali', 'V', 'Y', 'Z', 'Bos'
]) 
sequence_length = 30 
threshold = 0.60 


gorsel_harfler = {
    'C_alt': 'C',
    'G_yumusak': 'G',
    'I_noktasiz': 'I',
    'I_noktali': 'i',
    'O_noktali': 'O',
    'S_alt': 'S',
    'U_noktali': 'U'
}

sequence = []
sentence = [] 

ardisik_sayac = 0       
anlik_tahmin = ""       
izin_verilen_harf = ""  

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)
        
        el_gorunuyor_mu = results.left_hand_landmarks or results.right_hand_landmarks
        
        if el_gorunuyor_mu:
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:] 
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                guven_skoru = res[np.argmax(res)]
                
                if guven_skoru > threshold: 
                    gercek_harf = actions[np.argmax(res)]
                    
                    # 1. KARARLILIK FILTRESI
                    if gercek_harf == anlik_tahmin:
                        ardisik_sayac += 1
                    else:
                        anlik_tahmin = gercek_harf
                        ardisik_sayac = 0

                    # 2. Ekranda göstereceğimiz harfi sözlükten al (Yoksa kendisini kullan)
                    ekran_harfi = gorsel_harfler.get(gercek_harf, gercek_harf)

                    # 3. EKRANA YAZMA ONAYI
                    if ardisik_sayac >= 15:
                        # HARF "BOS" DEGILSE CUMLEYE EKLE!
                        if gercek_harf != 'Bos' and gercek_harf != izin_verilen_harf:
                            sentence.append(ekran_harfi)
                            izin_verilen_harf = gercek_harf 
                            
                    # 4. Sol Üst Köşe Tahmin Kutusu (Bos ise gizle)
                    if gercek_harf != 'Bos':
                        cv2.rectangle(image, (0,0), (250, 40), (245, 117, 16), -1)
                        cv2.putText(image, f'{ekran_harfi} ({guven_skoru*100:.0f}%)', (15, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    else:
                        # Bos yapildiysa ekranda Bos yazmasin, yesil ile Dinleniyor yazsin
                        cv2.putText(image, 'Dinleniyor...', (15, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    ardisik_sayac = 0 
                    cv2.putText(image, 'Tahmin Bekleniyor...', (15, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            sequence = []
            ardisik_sayac = 0
            anlik_tahmin = ""
            izin_verilen_harf = "" 
            cv2.putText(image, 'El Bekleniyor...', (15, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        if len(sentence) > 10:
            sentence = sentence[-10:]

        cv2.rectangle(image, (0, 640), (1280, 720), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (15, 690), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

        cv2.imshow('TID Canli Test - Huseyin Karaarslan', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()