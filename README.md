# Predictive-Modeling-and-Content-Segmentation-of-Netflix-Shows-Using-Data-Mining-Techniques
The project aims to analyze the Netflix dataset using classification and clustering to uncover patterns in content types, release years, and viewer behavior. Key features include show type, title, director, cast, country, date added, release year, rating, duration, genre, description, popularity, budget, and revenue.

# Predictive Modeling and Content Segmentation of Netflix Shows Using Data Mining Techniques

**Authors:**  
- Parth Deshmukh, BITS Pilani Dubai Campus, f20230399@dubai.bitspilani.ac.in  
- Kavyasree Nunna, BITS Pilani Dubai Campus, f20230072@dubai.bitspilani.ac.in  

---

## **Abstract**

This project explores the Netflix dataset using **classification and clustering** techniques to uncover patterns in content categorization, release years, and viewer behavior. The dataset includes features such as show type, title, director, cast, country, date added, release year, rating, duration, genre, description, popularity, budget, and revenue.  

---

## **Introduction**

Netflix and other streaming services generate huge amounts of data daily, including movie metadata, user ratings, and viewing behavior. This project applies **data preprocessing**, **classification**, and **clustering** techniques to extract meaningful insights from the dataset. Raw data is cleaned to remove inconsistencies and missing values before analysis.  

---

## **Dataset Description**

The dataset contains:

- Movie/TV show metadata (title, type, director, cast, country, language, release year)  
- Audience engagement metrics (ratings, vote counts, popularity)  
- Financial information (budget, revenue, ROI)  
- Content descriptors (genre, description, duration)

The dataset supports **classification** (predicting genres, ratings, or user preferences) and **clustering** (grouping similar content by thematic or financial features).  

---

## **Methodology**

**Workflow:**

1. **Data Ingestion:** Load CSV/Excel files and update periodically.  
2. **Preprocessing:**  
   - Remove duplicates  
   - Handle missing values (imputation)  
   - Convert types (e.g., budget, rating → numeric)  
   - Feature engineering (e.g., primary genre, ROI)  
   - Encode categorical variables (one-hot/label encoding)  
   - Normalize numeric features  

3. **Modeling:**  
   - **Classification:** Predict movie success (Hit, Average, Flop) using Decision Tree, KNN, Naive Bayes, SVM, and a custom rule-based classifier.  
   - **Clustering:** Group movies using K-means, Agglomerative, DBSCAN, Spectral, and custom rule-based clustering based on financial and popularity metrics.  

4. **Evaluation:**  
   - Classification: Accuracy, Precision, Recall, F1-score, Confusion Matrix  
   - Clustering: WCSS, Silhouette Coefficient, Davies-Bouldin Index, Dunn Index  

5. **Visualization:** Graphs and charts for model performance and clustering results.  

---

## **Preprocessing Details**

- Missing values filled using **median** (numeric) or **'Unknown'** (categorical)  
- Feature engineering for **primary genre** and **ROI**  
- Scaling/normalization ensures uniform feature impact  

---

## **Classification Details**

- Success classes defined by ROI:  
  - Hit → ROI > 50%  
  - Average → ROI 10–50%  
  - Flop → ROI < 10%  
- SVM achieved the **highest accuracy, precision, recall, and F1-score**.  
- Naive Bayes performed the weakest due to feature independence assumptions.  

---

## **Clustering Details**

- Features: budget, revenue, ROI, popularity, vote_average, vote_count  
- PCA reduces features to 2D for visualization  
- DBSCAN performed best due to noise handling and irregular cluster shapes  
- Custom clusters identify:  
  - Indie Hits (low-budget, high ROI)  
  - Blockbusters (high-budget, high revenue)  

---

## **Results**

- **Classification:** SVM consistently outperforms other models; Naive Bayes underperforms  
- **Clustering:** DBSCAN produces the most meaningful clusters based on multiple metrics  
- Visualization includes accuracy, precision, recall, F1-score graphs, confusion matrices, WCSS, Silhouette, Dunn Index, and DBI charts  

---

## **Software & Hardware Requirements**

**Software:**  
- Python 3.8+  
- Libraries: pandas, NumPy, scikit-learn, matplotlib, seaborn  
- VS Code or any IDE  

**Hardware:**  
- Processor: Intel Core i5/i7 or AMD Ryzen 5/7  
- RAM: 8 GB minimum (16 GB recommended)  
- Storage: 256 GB SSD  
- OS: Windows 10/11, macOS, or Linux  

---

## **Conclusion**

The project demonstrates that **robust preprocessing combined with advanced classification and clustering** leads to meaningful insights in Netflix content segmentation and predictive modeling. SVM and DBSCAN were the most effective techniques, offering guidance for future recommendation systems and content strategy analysis.  

---

## **References**

1. A. Jain, M. Murty, and P. Flynn, “Data Clustering: A Review,” ACM Computing Surveys, 1999.  
2. L. Rokach and O. Maimon, “Clustering Methods,” Data Mining and Knowledge Discovery Handbook, Springer, 2010.  
3. T. Zhang, R. Ramakrishnan, and M. Livny, “BIRCH: An Efficient Data Clustering Method for Very Large Databases,” ACM SIGMOD Record, 1996.  
4. X. Xu, M. Ester, H.-P. Kriegel, and J. Sander, “A Distribution‑Based Clustering Algorithm for Mining in Large Spatial Databases,” 1998.  
5. J. MacQueen, “Some Methods for Classification and Analysis of Multivariate Observations,” 1967.  
6. P. J. Rousseeuw, “Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis,” 1987.  
7. D. L. Davies and D. W. Bouldin, “A Cluster Separation Measure,” IEEE TPAMI, 1979.  
8. M. Halkidi, Y. Batistakis, and M. Vazirgiannis, “On Clustering Validation Techniques,” 2001.  
9. Y. Koren, R. Bell, and C. Volinsky, “Matrix Factorization Techniques for Recommender Systems,” IEEE Computer, 2009.  
10. A. Sharma and V. Singh, “Hybrid Clustering and Recommendation Approaches for Large‑Scale Multimedia Catalogs,” TKDD, 2024.  


