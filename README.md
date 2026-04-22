# ✈️ Rational Flight Dispatcher AI

Bu proje, İstinye Üniversitesi **Principles of AI** dersi final ödevi kapsamında geliştirilmiştir. Bir havalimanındaki uçak trafiğini yöneten, rasyonel bir yapay zeka ajanını simüle eder.

## 🧠 Proje Sütunları (Foundational Pillars)

Proje, dersin gereksinimlerini karşılayan üç temel teknik üzerine inşa edilmiştir:

1.  **Logic (Mantık):** `logic_engine.py` içerisinde **Modus Ponens** çıkarım kuralları kullanılarak uçuş öncelikleri belirlenir. (Örn: Yakıt kritikse ve hava fırtınalıysa acil iniş önceliği atanır.)
2.  **The Math of AI (Matematik):** `math_model.py` içerisinde uçak koordinatları **Lineer Cebir** (matris dönüşümleri) ile hesaplanır ve rüzgarın iniş üzerindeki etkisi **Olasılık** teorisi ile modellenir.
3.  **Optimization (Optimizasyon):** `optimizer.py` içerisinde, toplam bekleme süresini minimize etmek için **Hill Climbing** algoritması kullanılarak en ideal iniş sırası oluşturulur.

## 🛠️ Kurulum ve Çalıştırma

Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1. **Gereksinimleri Yükleyin:**
   ```bash
   pip install -r requirements.txt
