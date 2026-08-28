# Clustering Algorithms on Gas Sensor Drift Data

*Tutorial 12 — comparing K-Means, DBSCAN, Mutual k-NN Graph Clustering, and Affinity Propagation on the UCI Gas Sensor Array Drift Dataset.*

**Author:** Aman Kumar (Roll No. 230107012)
**Notebook:** `T12_230107012.ipynb`

---

##  Overview

This project applies and compares four unsupervised clustering algorithms on a high-dimensional chemical sensor dataset:

- **K-Means** — used as the baseline
- **DBSCAN** — density-based clustering
- **Mutual k-NN Graph Clustering** — graph-based clustering via connected components
- **Affinity Propagation** — exemplar-based clustering

Each algorithm is evaluated and visualized in both the original feature space (post-scaling) and in **PCA-reduced 2D space**, using three internal cluster-validity metrics: **Silhouette Score**, **Davies–Bouldin Index**, and **Calinski–Harabasz Score**.

---

##  Dataset

**UCI Gas Sensor Array Drift Dataset** ([archive.ics.uci.edu/dataset/270](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset))

- 13,910 measurements from 16 metal-oxide (MOX) gas sensors
- 128 features per reading (8 features × 16 sensors: steady-state + dynamic response)
- 6 gas classes: Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone, Toluene
- Collected across 10 time-batches spanning 36 months

The notebook expects a single merged CSV, **`gas_sensor_all_batches.csv`**, with columns:

```
batch, gas_label, gas_name, feature_1, feature_2, ..., feature_128
```

> The dataset file itself is **not included** in this repo (large, and originally distributed as separate `batch1.dat`–`batch10.dat` files by UCI). Download it from the link above, merge/label the batches into the column format shown, and place the CSV in the project root before running the notebook.

---

##  Project Structure

```
.
├── T12_230107012.ipynb        # Main notebook (all analysis + models)
├── gas_sensor_all_batches.csv # Dataset (you add this — see Dataset section)
├── requirements.txt           # Python dependencies
└── README.md
```

---

##  Tech Stack

- **Language:** Python 3.11
- **Environment:** Jupyter Notebook
- **Libraries** (exact versions used in development):

| Library | Version |
|---|---|
| pandas | 2.1.4 |
| numpy | 1.26.4 |
| matplotlib | 3.8.0 |
| seaborn | 0.12.2 |
| scikit-learn | 1.2.2 |
| scipy | 1.11.4 |
| missingno | 0.5.2 |

---

##  Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add the dataset
#    Place gas_sensor_all_batches.csv (see Dataset section) in the project root

# 5. Launch Jupyter
jupyter notebook T12_230107012.ipynb
```

Run the cells **top to bottom, in order**. A few cells were originally run out of sequence during development (see [Known Limitations](#-known-limitations--todo)), so for consistent results use **Kernel → Restart & Run All** rather than running cells individually out of order.

---

##  Notebook Walkthrough

**Section 1 — Imports**
Loads pandas, numpy, matplotlib, seaborn, `sklearn.datasets`, and `missingno`.

**Section 2 — Data Loading & Preprocessing**
Loads the CSV into `aman_df`, inspects shape/dtypes with `.head()` / `.tail()`, and checks for missing values and duplicate rows.

**Section 3 — Exploratory Data Analysis**
- Drops highly correlated features (correlation threshold > 0.9) to reduce redundancy
- KDE plots per remaining numeric feature
- Correlation heatmap and pairplot on a sample of points
- IQR-based outlier removal
- Feature scaling with `StandardScaler`
- PCA scree plot to inspect explained variance

**Section 4 — Baseline Clustering (K-Means)**
Fits K-Means (k = 3), plots the Elbow curve (k = 1–10) to sanity-check the choice of k, evaluates with all three metrics, and visualizes clusters both in raw feature space and in PCA space.

**Section 5 — DBSCAN**
Uses a k-distance graph (k = 4) to estimate a good `eps`, sets `min_samples = 2 × n_features`, fits DBSCAN, counts clusters/noise points, visualizes clusters (noise marked separately) in PCA space, evaluates metrics, and sweeps several `eps` values for comparison.

**Section 6 — Mutual k-NN Graph Clustering**
Builds a k-NN graph (`NearestNeighbors`), keeps only **mutual** neighbor edges, extracts clusters as **connected components** (`scipy.sparse.csgraph.connected_components`), visualizes in PCA space, and sweeps k ∈ {3, 5, 10}.

**Section 7 — Affinity Propagation**
Fits `AffinityPropagation`, retrieves exemplars (`cluster_centers_indices_`), and explores tuning the `preference` parameter around the median/std of pairwise similarities. *(Present in the notebook but not executed in the saved run — see below.)*

**Section 8 — Comparison of All Algorithms**
Intended to summarize all four algorithms side by side across the three metrics, on both original and PCA-reduced data. *(Left as a scaffold — not filled in yet.)*

---

##  Results (from the saved run)

**Baseline vs. graph/density methods:**

| Algorithm | Configuration | Clusters Found | Noise Points | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ |
|---|---|---|---|---|---|---|
| K-Means (baseline) | k = 3 | 3 | — | 0.209 | 1.651 | 3588.79 |
| DBSCAN | eps = 1.0, min_samples = 5 | 217 | 2,114 | 0.331 | 0.878 | 654.98 |
| Mutual k-NN Graph | k = 5 | 1,945 | — | 0.042 | 0.929 | 323.98 |
| Affinity Propagation | default params | *not executed in saved run* | — | — | — | — |

> DBSCAN note: the notebook's DBSCAN-fitting cell is set to `eps=1.5, min_samples=60`, but the metrics captured in the saved run correspond to `eps=1.0, min_samples=5` (matching the sweep below) — a sign the cells were run out of order at some point. Re-run top-to-bottom for a fully consistent set of numbers.

**DBSCAN — eps sweep** (`min_samples = 5`):

| eps | Clusters | Noise Points | Silhouette |
|---|---|---|---|
| 0.5 | 109 | 7,026 | 0.497 |
| 0.8 | 201 | 3,691 | 0.421 |
| 1.0 | 217 | 2,114 | 0.331 |
| 1.2 | 155 | 1,192 | 0.073 |
| 1.5 | 98 | 483 | −0.082 |

**Mutual k-NN Graph — k sweep:**

| k | Clusters | Silhouette |
|---|---|---|
| 3 | 4,824 | 0.106 |
| 5 | 1,945 | 0.042 |
| 10 | 572 | 0.006 |

**Takeaway:** DBSCAN (low eps) gives the best Silhouette scores but at the cost of very fragmented clusters or high noise counts; K-Means gives the most "usable" small number of clusters but a modest Silhouette; the mutual k-NN graph approach tends to over-segment the data at low k.

---

##  Evaluation Metrics

- **Silhouette Score** (−1 to 1, higher is better) — measures how well-separated and cohesive clusters are.
- **Davies–Bouldin Index** (lower is better) — average similarity between each cluster and its most similar one; lower means better separation.
- **Calinski–Harabasz Score** (higher is better) — ratio of between-cluster to within-cluster dispersion.

---

##  Known Limitations / TODO

- Affinity Propagation (Section 7) and the final cross-algorithm comparison (Section 8) are scaffolded but not executed/completed in the saved notebook.
- The DBSCAN fitting cell's parameters (`eps=1.5, min_samples=60`) don't match the metrics captured immediately after it — re-run sequentially to resolve.
- `LinearDiscriminantAnalysis` and `TSNE` are imported in Section 3 but not currently used anywhere.
- The dataset is not version-controlled in this repo — consider a `data/` folder with a `.gitignore` entry, or Git LFS, plus a small download script.

---

##  References

- UCI Machine Learning Repository — [Gas Sensor Array Drift Dataset](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset)
- [scikit-learn documentation](https://scikit-learn.org/stable/)

---

## 📄 License

No license file is currently included. Add a `LICENSE` file (e.g., MIT) if you plan to make this repository public.
