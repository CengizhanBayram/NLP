import matplotlib.pyplot as plt 
# Ortalama hesaplama fonksiyonu
def mean(values):
    return sum(values) / len(values)

# Verileri normalize etme fonksiyonu (min-max normalizasyonu)
def normalize(X):
    min_x = min(X)
    max_x = max(X)
    return [(x - min_x) / (max_x - min_x) for x in X]

# Polinomsal özellikler oluşturma fonksiyonu
def polynomial_features(X, degree):
    poly_X = []
    for x in X:
        poly_features = [x**i for i in range(degree + 1)]  # x^0, x^1, ..., x^degree
        poly_X.append(poly_features)
    return poly_X

# Tahmin fonksiyonu (beta ve X kullanarak y'yi tahmin etme)
def predict(X, beta):
    y_pred = []
    for x in X:
        y = sum([beta[i] * x[i] for i in range(len(beta))])
        y_pred.append(y)
    return y_pred

# Polinomsal regresyon fonksiyonu
def polynomial_regression(X, y, degree, learning_rate=0.001, epochs=10000, cost_threshold=0.001):
    # Eğitim örneklerinin sayısı
    m = len(y)

    # Veriyi normalize ediyoruz
    X_normalized = normalize(X)

    # Polinomsal özellikler oluşturuluyor
    poly_X = polynomial_features(X_normalized, degree)

    # Başlangıç parametreleri (beta)
    beta = [0] * (degree + 1)  # Her polinom derecesi için bir beta

    # Gradient descent ile parametreleri güncelle
    for epoch in range(epochs):
        # Tahmin edilen y değerleri
        y_pred = predict(poly_X, beta)

        # Hatalar (y_pred - y)
        error = [y_pred[i] - y[i] for i in range(m)]

        # Maliyet fonksiyonu (mean squared error)
        cost = sum([e**2 for e in error]) / (2 * m)

        # Gradient descent güncellemeleri
        beta_deriv = [0] * (degree + 1)
        for j in range(degree + 1):
            beta_deriv[j] = sum([error[i] * poly_X[i][j] for i in range(m)]) / m

        # Parametre güncelleme
        for j in range(degree + 1):
            beta[j] = beta[j] - learning_rate * beta_deriv[j]

        # Her 100 iterasyonda maliyeti yazdır ve durma kontrolü yap
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}: Cost = {cost}, beta = {beta}")

        # Eğer cost eşik değerden düşükse, eğitimi durdur
        if cost < cost_threshold:
            print(f"Cost değeri {cost_threshold}'nin altına düştü. Eğitim durduruluyor.")
            break
    
    return beta

# Örnek veri
X = [1, 2, 3, 4, 5,6,7,8,9,10,11]
y = [1, 4, 9, 16, 25,33,44,55,66,77,88]  # Bu, y = x^2'e yakın bir ilişki

# Polinomsal regresyon modelini eğit
degree = 2  # 2. dereceden (quadratic) bir model
beta = polynomial_regression(X, y, degree, learning_rate=0.001, epochs=100000, cost_threshold=0.001)

# Sonuçları yazdır
print(f"Model: y = {' + '.join([f'{b:.2f}x^{i}' for i, b in enumerate(beta)])}")
poly_X = polynomial_features(normalize(X), degree)

 # Son tahminler (modelin çıkardığı sonuçlar)
final_predictions = predict(poly_X, beta)
# Grafik çizimi
plt.scatter(X, y, color='blue', label='Gerçek Değerler')  # Gerçek değerler
plt.plot(X, final_predictions, color='red', label='Tahmin Edilen Değerler')  # Modelin tahmin ettiği değerler
plt.title(f"Polinomsal Regresyon (Derece {degree})")
plt.xlabel("X Değerleri")
plt.ylabel("Y Değerleri")
plt.legend()
plt.show()

