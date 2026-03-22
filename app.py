from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app) # to connect to React

#Load Model
model = load_model("regression_lstm_yelp.h5", compile=False)

#Load Tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

#Rest API Endpoint

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    try:
        # 1. React'ten gelen veriyi al
        data = request.get_json()
        
        # DEDEKTİF KODU 1: Gelen veriyi terminale yazdır
        print(f"\n--- YENİ İSTEK GELDİ ---")
        print(f"1. React'ten Gelen Ham Veri: {data}")

        # Verinin içinden 'review' anahtarını çek
        user_review = data.get('review', '')

        if not user_review or str(user_review).strip() == "":
            return jsonify({'error': 'Lütfen geçerli bir yorum girin.'}), 400

        # 2. Metni sayılara çevir (Tokenizer)
        encoded = tokenizer.texts_to_sequences([user_review])
        
        # DEDEKTİF KODU 2: Tokenizer kelimeleri tanımış mı?
        print(f"2. Tokenizer Çıktısı (Sayısal Dizi): {encoded}")
        
        # Eğer dizi boşsa [[]] (Yani tokenizer kelimelerin hiçbirini sözlükte bulamadıysa)
        if len(encoded[0]) == 0:
            print("⚠️ UYARI: Tokenizer bu kelimelerin hiçbirini tanımadı!")

        # Uzunluğu 200'e sabitle
        processed_input = pad_sequences(encoded, maxlen=200)
        
        # 3. Model Tahmini
        prediction = model.predict(processed_input)[0][0]
        print(f"3. Modelin Ham Çıktısı (0.0 - 1.0 arası): {prediction}")

        # --- YENİ: MATEMATİKSEL ESNETİCİ (AMPLIFIER) ---
        # Model 0.5'e (nötr) yakın değerler veriyorsa, bu değerleri uçlara doğru itiyoruz.
        # prediction_shifted: -0.5 ile +0.5 arası bir değer olur
        prediction_shifted = prediction - 0.5
        
        # Değeri 1.8 kat güçlendiriyoruz (Bu katsayıyı artırarak daha da uçlara itebilirsin)
        amplified_prediction = prediction_shifted * 1.8 
        
        # Değeri tekrar 0 ile 1 arasına çekiyoruz (ve sınırları taşırmasını engelliyoruz)
        final_probability = max(0.0, min(1.0, amplified_prediction + 0.5))

        # 4. SKORU 1 İLE 5 ARASINA ÖLÇEKLEME (React arayüzüne uyum için)
        # 0.0 -> 1 puan, 1.0 -> 5 puan olacak şekilde matematiğe döküyoruz
        score_1_to_5 = 1 + (prediction * 4)
        print(f"4. React'e Gönderilen Puan: {score_1_to_5:.1f}\n")

        # React'in beklediği formatta (response.data.score) gönder
        # JSON hatasını önlemek için değerleri açıkça float() ile sarmalıyoruz
        # Aksi takdirde numpy.float32 tipi JSON'a dönüştürülemez.
        return jsonify({
            'score': float(round(score_1_to_5, 1)),
            'raw_score': float(prediction)
        })

    except Exception as e:
        # Hata mesajını daha detaylı görmek için
        import traceback
        print(traceback.format_exc()) 
        return jsonify({'error': str(e)}), 500

#to strat Flask API

if __name__ == '__main__':
    app.run(debug=True, port=5000)