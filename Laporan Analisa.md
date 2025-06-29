# 🛒 Analisis Perilaku Belanja Konsumen: Proyek Clustering Data Mining


## 📝 Deskripsi Proyek

Proyek ini merupakan tugas besar mata kuliah *Data Mining* yang berfokus pada analisis perilaku belanja konsumen. Kami menggunakan teknik clustering (*K-Means* dan *Agglomerative Clustering*) untuk mengelompokkan pelanggan berdasarkan data historis pembelian mereka. Tujuannya adalah untuk mengidentifikasi segmen pelanggan yang berbeda, yang dapat membantu strategi pemasaran, penawaran produk, dan layanan yang lebih personal.

Dataset yang digunakan adalah shopping_behavior_updated.csv, yang berisi data pembelian, demografi, dan preferensi dari berbagai pelanggan.

## 👥 Anggota Tim

Proyek ini dikerjakan oleh kelompok kami yang terdiri dari:

-   *Dipo Arya Mukti*
-   *Misbachuddin*
-   *Muhammad Iqbal*
-   *Nouver Effridho*
-   *Sultan Saukah Ibrahim*

## 💻 Teknologi yang Digunakan

Proyek ini dibangun menggunakan *Python* di lingkungan *Google Colab* dengan library sebagai berikut:

-   *Pandas & NumPy:* Untuk manipulasi dan analisis data.
-   *Matplotlib & Seaborn:* Untuk visualisasi data eksplorasi (EDA) dan hasil clustering.
-   *Scikit-learn:* Untuk preprocessing data (LabelEncoder, StandardScaler, PCA), algoritma clustering (KMeans, AgglomerativeClustering), dan evaluasi model (silhouette_score).
-   *SciPy:* Untuk visualisasi dendrogram dalam Hierarchical Clustering.

## 🛠 Instalasi & Penggunaan

Ikuti langkah-langkah mudah di bawah ini untuk menjalankan kode di *Google Colab*:

1.  *Clone Repository:*
    bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    
2.  *Upload Dataset:* Unggah file shopping_behavior_updated.csv ke dalam direktori Google Colab Anda.
3.  *Jalankan Notebook:* Buka file notebook (.ipynb) di Google Colab dan jalankan setiap cell secara berurutan.

## 📖 Penjelasan Kode (Langkah demi Langkah)

Berikut adalah penjelasan terperinci dari setiap cell kode yang ada di notebook Anda:

### **Cell 1: Impor Library**

python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

## *Penjelasan:* Cell ini mengimpor semua *library Python* yang dibutuhkan untuk keseluruhan proyek, termasuk:
-   pandas dan numpy untuk manipulasi data 📊
-   matplotlib dan seaborn untuk visualisasi data 📈
-   Berbagai modul dari scikit-learn untuk preprocessing, clustering, dan reduksi dimensi ✨

### **Cell 2: Memuat dan Memeriksa Dataset**

python
file_path = 'shopping_behavior_updated.csv'
data = pd.read_csv(file_path)
data.head()
data
`
## *Penjelasan:*
Kode ini membaca dataset shopping_behavior_updated.csv ke dalam sebuah DataFrame Pandas. Perintah data.head() menampilkan 5 baris pertama data untuk pemeriksaan awal, sementara data menampilkan keseluruhan DataFrame, memastikan data berhasil dimuat.

### **Cell 3: Pra-pemrosesan Data (bagian 1)**
python
# Data Preprocessing
data.drop('Customer ID', axis=1, inplace=True)

# There is no missing data

# Label Encoding for binary columns
le = LabelEncoder()
binary_columns = ['Gender', 'Subscription Status', 'Discount Applied', 'Promo Code Used']
for col in binary_columns:
    data[col] = le.fit_transform(data[col])

numerical_columns = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
categorical_columns = ['Item Purchased', 'Category', 'Location', 'Size', 'Color', 'Season',
                       'Shipping Type', 'Payment Method', 'Frequency of Purchases']

## *Penjelasan:*
Pada tahap ini, data dipersiapkan untuk analisis:
1. Kolom 'Customer ID' dihapus karena merupakan ID unik dan tidak relevan untuk proses clustering.
2. Kolom biner ('Gender', 'Subscription Status', dll.) diubah menjadi nilai numerik (0 dan 1) menggunakan LabelEncoder.
3. Kolom-kolom dipisahkan menjadi tiga kelompok (numerical, categorical, dan yang sudah biner) untuk pemrosesan lebih lanjut.

### **Cell 4: Visualisasi Korelasi**
python
corr = data.loc[:, ~data.columns.isin(categorical_columns)].corr() #exclude categorical_columns

plt.subplots(figsize=(10,8))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
plt.show()

## *Penjelasan:*
Heatmap korelasi ini dibuat untuk memahami hubungan antar fitur numerik dan biner. Plot ini menunjukkan seberapa kuat hubungan linear antar variabel, yang dapat memberikan wawasan awal sebelum pemodelan.

### **Cell 5: Pra-pemrosesan Data (Bagian 2)**
python
# One-Hot Encoding for non-binary categorical columns
data = pd.get_dummies(data, columns=categorical_columns)

## *Penjelasan:*
Semua kolom kategorikal non-biner (seperti 'Category', 'Location', dll.) diubah menjadi representasi numerik menggunakan One-Hot Encoding. Ini mengubah setiap kategori unik menjadi kolom biner baru (0 atau 1), yang diperlukan untuk algoritma clustering.

### **Cell 6: Standarisasi Data**
python
# Standard Scaler for numerical columns
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

## *Penjelasan:*
Variabel numerik ('Age', 'Purchase Amount', dll.) memiliki skala yang berbeda. StandardScaler digunakan untuk menstandarisasi kolom-kolom ini agar memiliki mean 0 dan deviasi standar 1, sehingga setiap fitur memiliki bobot yang sama dalam perhitungan jarak oleh algoritma clustering.

### **Cell 7: Menampilkan Data yang sudah di proses**
python
# Display the first few rows of the preprocessed dataset
data

## *Penjelasan:*
Cell ini menampilkan DataFrame yang sudah diproses secara menyeluruh (setelah di-encoding dan distandarisasi), menunjukkan bagaimana data telah diubah dan siap untuk pemodelan

### **Cell 8 & 9: Ekspor dan Muat Ulang Dataset**
python
# Export the DataFrame to a CSV file
data.to_csv("shopping_behavior_updated1.csv", index=False)

# Load the dataset
file_path = 'shopping_behavior_updated1.csv'
data = pd.read_csv(file_path)

## *Penjelasan:*
Data yang sudah diproses disimpan ke file CSV baru (shopping_behavior_updated1.csv) untuk kemudahan akses. Kemudian, file tersebut langsung dimuat ulang, yang mungkin bertujuan untuk memastikan data yang akan dianalisis adalah versi yang sudah bersih dan siap.

### **Cell 10: Visualisasi Fitur**
python
plt.figure(figsize=(6, 4))
sns.barplot(x='Gender', y='Item Purchased_Backpack', data=data, ci=None, palette='viridis')
plt.title("Impact of Gender on Item Purchased_Backpack")
plt.xticks(rotation=45)
plt.show()

## *Penjelasan:*
Diagram batang ini dibuat untuk melihat hubungan antara jenis kelamin (Gender) dan pembelian 'Backpack' secara spesifik. Ini adalah salah satu contoh dari visualisasi data eksplorasi (EDA) untuk mendapatkan wawasan dari fitur-fitur yang sudah di-encoding.

### **Cell 11: Menghitung Silhouette Score**
python
from sklearn.metrics import silhouette_score

def calculate_silhouette_scores(data, range_clusters):
    silhouette_scores = []
    for cluster in range_clusters:
        kmeans = KMeans(n_clusters=cluster, init='k-means++', n_init=10)
        kmeans.fit(data)
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(data, labels)
        silhouette_scores.append(silhouette_avg)
    return silhouette_scores

range_clusters = range(2, 21)
silhouette_scores = calculate_silhouette_scores(data, range_clusters)

# Plotting the silhouette scores
plt.plot(range_clusters, silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

## *Penjelasan:*
Cell ini menggunakan Silhouette Score untuk menentukan jumlah cluster yang optimal. Fungsi calculate_silhouette_scores menghitung skor rata-rata untuk setiap jumlah cluster dalam rentang 2 hingga 20. Semakin tinggi skornya, semakin baik pemisahan antar cluster. Plot yang dihasilkan membantu visualisasi skor untuk setiap jumlah cluster.

### **Cell 12: Menentukan Jumlah Cluster Optimal**
python
# Find the optimal number of clusters
optimal_clusters = range_clusters[np.argmax(silhouette_scores)]
print("Optimal Number of Clusters:", optimal_clusters)

# Highlight the optimal point on the plot
plt.scatter(optimal_clusters, max(silhouette_scores), color='red', marker='x', label='Optimal Clusters')
plt.legend()
plt.show()

## *Penjelasan:*
Berdasarkan plot di cell sebelumnya, kode ini secara otomatis menemukan jumlah cluster yang memiliki Silhouette Score tertinggi. Hasilnya menunjukkan bahwa jumlah cluster optimal adalah 2. Titik ini kemudian ditandai pada grafik untuk memudahkan identifikasi.

### **Cell 13 & 17: K-Means Clustering**
python
# Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# Train K-Means on the PCA-transformed data
kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(data_pca)
# ... (Cells 14, 15, 16, 17)
kmeans_silhouette = silhouette_score(data, kmeans.labels_).round(2)
print(f"Silhouette Score: {kmeans_silhouette}")

## *Penjelasan:*
1. Reduksi Dimensi (PCA): PCA (Principal Component Analysis) digunakan untuk mengurangi dimensi data dari puluhan fitur menjadi hanya 2 komponen utama. Ini diperlukan untuk memvisualisasikan hasil clustering di ruang 2D.
2. Penerapan K-Means: Model K-Means dilatih dengan 2 cluster (sesuai hasil optimal di cell 12).
3. Evaluasi: Atribut model seperti inertia_ (sum of squared distances), cluster_centers_, dan n_iter_ ditampilkan. Silhouette Score untuk model K-Means dihitung dan dicetak sebagai metrik evaluasi.

### **Cell 18: Visualisasi K-Means**
python
# Visualize the clusters
plt.scatter(data_pca[y_kmeans == 0, 0], data_pca[y_kmeans == 0, 1], s=100, c='green', label='Cluster 1')
plt.scatter(data_pca[y_kmeans == 1, 0], data_pca[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')

# Plot the centroids in the PCA space
centers_pca = kmeans.cluster_centers_
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], s=200, c='yellow', label='Centroids')

# Finalize the plot
plt.title('K-Means Clustering on PCA-transformed Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

## *Penjelasan:*
Plot sebar (scatter plot) ini memvisualisasikan hasil clustering K-Means di ruang 2D yang direduksi oleh PCA. Setiap titik data diwarnai sesuai cluster-nya, dan centroid (pusat cluster) ditandai dengan warna kuning. Visualisasi ini menunjukkan bagaimana pelanggan dikelompokkan.

### **Cell 19 & 23: Agglomerative Clustering**
python
import scipy.cluster.hierarchy as sch
# Create the dendrogram
dendrogram = sch.dendrogram(sch.linkage(data_pca, method="ward"))
# ... (Cells 20, 21, 22, 23)
# Visualize the clusters
plt.figure(figsize=(10, 7))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=y_hc, cmap='viridis', edgecolors='k')
plt.title('Clusters of customers')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

## *Penjelasan:*
1. Dendrogram: Dendrogram dibuat untuk memvisualisasikan proses penggabungan cluster dalam Hierarchical Clustering.
2. Agglomerative Clustering: Algoritma clustering kedua (AgglomerativeClustering) dilatih dengan n_clusters=2.
3. Evaluasi & Visualisasi: Silhouette Score dihitung untuk model ini. Hasil clustering kemudian divisualisasikan menggunakan plot sebar di ruang PCA, menunjukkan perbandingan hasil dengan K-Means.

### **Cell 24 & 28: Analisis Profil Cluster**
python
# Undo Standard Scaling
data_undo = data.copy()
data_undo[numerical_columns] = scaler.inverse_transform(data_undo[numerical_columns])
# ... (remaining code in cells 24-28)

## *Penjelasan:*
Tahap ini adalah yang terpenting, yaitu interpretasi hasil clustering.
1. Mengembalikan Skala: Data numerik dikembalikan ke skala aslinya (inverse_transform) agar mudah diinterpretasikan.
2. Penambahan Label: Label cluster dari K-Means ditambahkan ke DataFrame.
3. Analisis Statistik: Rata-rata dari setiap fitur untuk setiap cluster dihitung dan ditampilkan, memberikan gambaran karakteristik masing-masing kelompok pelanggan.
4. Visualisasi Fitur: Plot histogram untuk 'Age', 'Purchase Amount', dan 'Previous Purchases' menunjukkan distribusi fitur-fitur ini di setiap cluster.
5. Analisis Biner: Statistik seperti rata-rata dan persentase dihitung untuk setiap kolom biner, menunjukkan perbedaan perilaku antar cluster dalam hal langganan, diskon, dll.

## *Kesimpulan dan Hasil*
Berdasarkan analisis, kami berhasil mengelompokkan pelanggan menjadi 2 cluster utama. Analisis mendalam menunjukkan perbedaan yang signifikan antara kedua kelompok ini, yang dapat digunakan untuk strategi bisnis yang lebih efektif.
