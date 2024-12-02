import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Memuat data
data1_path = "Dataset.xlsx"
data2_path = "rating_kendaraan.xlsx"

data1 = pd.read_excel(data1_path)
data2 = pd.read_excel(data2_path)

# Penyesuaian nama kolom sesuai dataset Anda
data1.columns = [
    'Nama_Kendaraan', 'Kapasitas_Koper', 'Kapasitas_Penumpang', 'Harga',
    'Vendor_Rating', 'Review', 'Order', 'Fitur1', 'Fitur2', 'Fitur3',
    'Harga_2', 'Hari', 'Lepas_Kunci', 'Sopir', 'Extra_Column'
]
data2.columns = ['Nama_Kendaraan', 'Rating']

# Menggabungkan dataset berdasarkan nama kendaraan
merged_data = pd.merge(data1, data2, on='Nama_Kendaraan', how='inner')

# Menggabungkan fitur deskripsi untuk analisis
merged_data['Combined_Features'] = (
    merged_data['Nama_Kendaraan'] + " " +
    merged_data['Kapasitas_Koper'].astype(str) + " " +
    merged_data['Kapasitas_Penumpang'].astype(str) + " " +
    merged_data['Harga'] + " " +
    merged_data['Fitur1'].fillna('') + " " +
    merged_data['Fitur2'].fillna('') + " " +
    merged_data['Fitur3'].fillna('')
)

# Menghitung TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(merged_data['Combined_Features'])

# Menghitung Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Fungsi rekomendasi
def recommend_kendaraan(kendaraan_name, cosine_sim=cosine_sim):
    idx = merged_data[merged_data['Nama_Kendaraan'] == kendaraan_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # 5 kendaraan teratas
    kendaraan_indices = [i[0] for i in sim_scores]
    return merged_data.iloc[kendaraan_indices][['Nama_Kendaraan', 'Rating']]