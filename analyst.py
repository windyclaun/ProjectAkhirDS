import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score

# Memuat data
data1_path = "Dataset.xlsx"
data2_path = "rating_kendaraan.xlsx"

data1 = pd.read_excel(data1_path)
data2 = pd.read_excel(data2_path)

# Penyesuaian nama kolom sesuai dataset
data1.columns = [
    'Nama_Kendaraan', 'Kapasitas_Koper', 'Kapasitas_Penumpang', 'Harga',
    'Vendor_Rating', 'Review', 'Order', 'Fitur1', 'Fitur2', 'Fitur3',
    'Harga_2', 'Hari', 'Lepas_Kunci', 'Sopir', 'Extra_Column'
]
data2.columns = ['Nama_Kendaraan', 'Rating']

# Menggabungkan dataset berdasarkan nama kendaraan
merged_data = pd.merge(data1, data2, on='Nama_Kendaraan', how='inner')

# Pembersihan dan Normalisasi Data
merged_data['Harga'] = merged_data['Harga'].str.replace('IDR ', '').str.replace('.', '').astype(float)

# Membersihkan dan konversi Review
merged_data['Review'] = merged_data['Review'].str.replace(r'\(.*\)', '', regex=True)  # Menghapus teks dalam tanda kurung
merged_data['Review'] = merged_data['Review'].str.replace(' Review', '')  # Menghapus kata 'Review'
merged_data['Review'] = merged_data['Review'].str.replace(' Order', '')  # Menghapus kata 'Order' jika ada
merged_data['Review'] = merged_data['Review'].str.replace(',', '.')  # Ganti koma dengan titik
merged_data['Review'] = merged_data['Review'].replace('', '0')  # Ganti nilai kosong dengan 0
merged_data['Review'] = merged_data['Review'].fillna('0')  # Mengisi nilai NaN dengan 0
merged_data['Review'] = merged_data['Review'].astype(float)  # Mengonversi menjadi float (bisa gunakan int jika diperlukan)
merged_data['Review'] = merged_data['Review'].astype(int)  # Mengonversi menjadi integer

# Membersihkan kolom 'Order' sebelum konversi
merged_data['Order'] = merged_data['Order'].str.replace(' Order', '')  # Menghapus kata 'Order'
merged_data['Order'] = merged_data['Order'].str.replace(' Review', '')  # Menghapus kata 'Review'
merged_data['Order'] = merged_data['Order'].str.replace('[^0-9]', '', regex=True)  # Menghapus karakter non-numerik (kecuali angka)
merged_data['Order'] = merged_data['Order'].fillna('0')  # Mengisi nilai NaN dengan '0'
merged_data['Order'] = merged_data['Order'].astype(int)  # Mengonversi menjadi integer

# Menentukan kolom numerik untuk normalisasi
numerical_features = ['Vendor_Rating', 'Harga', 'Review', 'Order', 'Rating']

# Memastikan bahwa hanya kolom numerik yang digunakan untuk normalisasi
merged_data[numerical_features] = merged_data[numerical_features].apply(pd.to_numeric, errors='coerce')

# Terapkan MinMaxScaler hanya pada kolom yang sudah dipastikan numerik
scaler = MinMaxScaler()
merged_data[numerical_features] = scaler.fit_transform(merged_data[numerical_features])

# Menggabungkan fitur deskripsi untuk analisis berbasis TF-IDF
merged_data['Combined_Features'] = (
    merged_data['Nama_Kendaraan'] + " " +
    merged_data['Kapasitas_Koper'].astype(str) + " " +
    merged_data['Kapasitas_Penumpang'].astype(str) + " " +
    merged_data['Harga'].astype(str) + " " +
    merged_data['Fitur1'].fillna('') + " " +
    merged_data['Fitur2'].fillna('') + " " +
    merged_data['Fitur3'].fillna('')
)

# Menghitung TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(merged_data['Combined_Features'])

# Menghitung Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Fungsi untuk rekomendasi berdasarkan harga dan keyword
def recommend_kendaraan_by_price_and_keyword(harga_min, harga_max, keyword, cosine_sim=cosine_sim):
    # Menyaring kendaraan berdasarkan harga yang diminta oleh user
    filtered_data = merged_data[(merged_data['Harga'] >= harga_min) & (merged_data['Harga'] <= harga_max)]
    
    # Filter kendaraan berdasarkan kecocokan dengan keyword di nama kendaraan dan fitur
    filtered_data = filtered_data[filtered_data['Combined_Features'].str.contains(keyword, case=False, na=False)]
    
    # Jika tidak ada kendaraan yang cocok, kembalikan pesan
    if filtered_data.empty:
        return "Tidak ada kendaraan yang sesuai dengan kriteria harga dan keyword."
    
    # Jika ada kendaraan yang cocok, hitung kesamaan cosine dan beri rekomendasi
    idx = filtered_data.index[0]  # Ambil kendaraan pertama yang cocok
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # 10 kendaraan teratas
    kendaraan_indices = [i[0] for i in sim_scores]
    
    return filtered_data.iloc[kendaraan_indices][['Nama_Kendaraan', 'Rating', 'Harga']]

# Menentukan relevansi berdasarkan Rating (misalnya rating > 4 dianggap relevan)
def is_relevant(row, threshold=4):
    return 1 if row['Rating'] > threshold else 0

# Fungsi untuk evaluasi menggunakan Precision, Recall, dan F1-Score
def evaluate_recommendations(predictions, ground_truth, threshold=4):
    # Membuat label relevansi untuk hasil rekomendasi
    predictions['relevant'] = predictions['Rating'].apply(is_relevant, threshold=threshold)
    
    # Membuat label relevansi untuk data ground truth
    ground_truth['relevant'] = ground_truth['Rating'].apply(is_relevant, threshold=threshold)
    
    # Menghitung Precision, Recall, dan F1-Score
    y_true = ground_truth['relevant']
    y_pred = predictions['relevant']
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return precision, recall, f1

# Input dari pengguna
harga_min = float(input("Masukkan harga minimum: "))
harga_max = float(input("Masukkan harga maksimum: "))
keyword = input("Masukkan keyword (misal nama kendaraan atau fitur): ")

# Mendapatkan rekomendasi
recommended_data = recommend_kendaraan_by_price_and_keyword(harga_min, harga_max, keyword)

# Ground truth (dari data2 yang sudah ada)
ground_truth_data = merged_data[['Nama_Kendaraan', 'Rating']]

# Evaluasi
precision, recall, f1 = evaluate_recommendations(recommended_data, ground_truth_data)

# Menampilkan rekomendasi dan hasil evaluasi
print(f"Rekomendasi berdasarkan harga dan keyword '{keyword}':")
print(recommended_data)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
