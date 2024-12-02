import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Membaca dataset
df = pd.read_csv('hasil_scraping_tiket.csv')

# Membersihkan dan mempersiapkan data

# Menghapus kolom yang tidak relevan (jika ada kolom kosong atau tidak terpakai)
df = df.dropna(axis=1, how='all')

# Menangani missing values pada kolom yang numerik
df['Harga (/hari)'] = df['Harga (/hari)'].str.replace('.', '')  # Menghapus titik (separator ribuan)
df['Harga (/hari)'] = df['Harga (/hari)'].str.replace(',', '.')  # Mengubah koma ke titik untuk float
df['Harga (/hari)'] = pd.to_numeric(df['Harga (/hari)'], errors='coerce')

# Menghapus baris dengan missing values pada kolom harga dan rating
df = df.dropna(subset=['Harga (/hari)', 'Rating Mobil'])

# Mengonversi kolom Rating Mobil menjadi numerik
df['Rating Mobil'] = pd.to_numeric(df['Rating Mobil'], errors='coerce')

# Mengonversi kolom lainnya jika perlu
df['Penumpang'] = pd.to_numeric(df['Penumpang'], errors='coerce')
df['Bagasi'] = pd.to_numeric(df['Bagasi'], errors='coerce')

# Menangani missing values pada kolom Penumpang dan Bagasi
df['Penumpang'].fillna(df['Penumpang'].median(), inplace=True)
df['Bagasi'].fillna(df['Bagasi'].median(), inplace=True)

# Fitur yang digunakan untuk KNN (Harga per hari, Penumpang, Bagasi, Rating Mobil)
X = df[['Harga (/hari)', 'Penumpang', 'Bagasi', 'Rating Mobil']]

# Label (kategori yang ingin diprediksi, misalnya 'Jenis Kendaraan')
# Di sini kita menggunakan rating mobil sebagai label
y = df['Rating Mobil']  # Menggunakan rating mobil sebagai label untuk klasifikasi

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalisasi atau Scaling data (karena KNN sensitif terhadap skala fitur)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Melatih model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Prediksi menggunakan data uji
y_pred = knn.predict(X_test)

# Evaluasi model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Menyimpan hasil prediksi ke dalam file CSV
df['Prediksi Rating Mobil'] = knn.predict(scaler.transform(df[['Harga (/hari)', 'Penumpang', 'Bagasi', 'Rating Mobil']]))
df.to_csv('prediksi_rating_mobil.csv', index=False)
