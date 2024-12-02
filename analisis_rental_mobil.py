import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca data dari file Excel
df = pd.read_excel('Dataset.xlsx')

# Tampilkan beberapa baris pertama untuk memastikan data terbaca dengan benar
print(df.head())

# Pembersihan Data
# Mengonversi harga ke dalam bentuk numerik (menghapus 'IDR' dan koma)
df['Harga'] = df['Field5'].str.replace('IDR ', '').str.replace('.', '').astype(float)

# Mengonversi rating menjadi numerik dengan mengganti koma menjadi titik
df['Rating'] = df['Rating'].str.replace(',', '.').astype(float)

# Membersihkan data di kolom 'Review' sebelum konversi
df['Review'] = df['Review'].str.replace(r'\(.*\)', '', regex=True)  # Menghapus teks dalam tanda kurung
df['Review'] = df['Review'].str.replace(' Review', '')  # Menghapus kata 'Review'
df['Review'] = df['Review'].str.replace(' Order', '')  # Menghapus kata 'Order' jika ada
df['Review'] = df['Review'].replace('', '0')  # Ganti nilai kosong dengan 0
df['Review'] = df['Review'].fillna(0)  # Mengisi nilai NaN dengan 0
df['Review'] = df['Review'].astype(int)  # Mengonversi menjadi integer

# Membersihkan data di kolom 'Order' sebelum konversi
df['Order'] = df['Order'].str.replace(' Order', '').str.replace('+', '').str.replace('.', '').fillna('0')  # Ganti NaN dengan '0'
df['Order'] = df['Order'].astype(int)  # Mengonversi menjadi integer

# Menampilkan info statistik dari data
print(df.describe())

# Melakukan analisis: Menampilkan korelasi antara rating, harga, dan review
correlation = df[['Rating', 'Harga', 'Review', 'Order']].corr()

# Menampilkan korelasi
print(correlation)

# Visualisasi Korelasi dengan Heatmap
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korelasi Antara Rating, Harga, Review, dan Order')
plt.show()

# Analisis Vendor dengan Rating Tertinggi
top_vendors = df.groupby('Vendor').agg({'Rating': 'mean'}).sort_values(by='Rating', ascending=False)

# Menampilkan 10 vendor dengan rating tertinggi
print(top_vendors.head(10))

# Plot distribusi harga
plt.figure(figsize=(10, 6))
sns.histplot(df['Harga'], kde=True, bins=20, color='blue')
plt.title('Distribusi Harga Mobil Sewa')
plt.xlabel('Harga (IDR)')
plt.ylabel('Frekuensi')
plt.show()

# Plot distribusi rating
plt.figure(figsize=(10, 6))
sns.histplot(df['Rating'], kde=True, bins=10, color='green')
plt.title('Distribusi Rating Vendor')
plt.xlabel('Rating')
plt.ylabel('Frekuensi')
plt.show()
