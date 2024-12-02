import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

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
merged_data['Review'] = merged_data['Review'].str.replace(r'\(.*\)', '', regex=True).fillna('0').astype(int)
merged_data['Order'] = merged_data['Order'].str.replace(' Order', '').str.replace('+', '').str.replace('.', '').fillna('0').astype(int)

# Normalisasi kolom numerik untuk analisis korelasi
features = ['Vendor_Rating', 'Harga', 'Review', 'Order', 'Rating']
scaler = MinMaxScaler()
merged_data[features] = scaler.fit_transform(merged_data[features])

# Visualisasi Korelasi
correlation = merged_data[features].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korelasi Antar Fitur')
plt.show()

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

# Fungsi rekomendasi berdasarkan TF-IDF
def recommend_kendaraan(kendaraan_name, cosine_sim=cosine_sim):
    idx = merged_data[merged_data['Nama_Kendaraan'] == kendaraan_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # 5 kendaraan teratas
    kendaraan_indices = [i[0] for i in sim_scores]
    return merged_data.iloc[kendaraan_indices][['Nama_Kendaraan', 'Rating']]

# Contoh rekomendasi
kendaraan_name = "Nama Kendaraan Contoh"  # Ganti dengan nama kendaraan yang ada di dataset
recommended = recommend_kendaraan(kendaraan_name)

print(f"Rekomendasi untuk '{kendaraan_name}':")
print(recommended)

# Visualisasi distribusi rating
plt.figure(figsize=(10, 6))
sns.histplot(merged_data['Rating'], kde=True, bins=10, color='green')
plt.title('Distribusi Rating Kendaraan')
plt.xlabel('Rating')
plt.ylabel('Frekuensi')
plt.show()
