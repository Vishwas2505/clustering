import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# App Title
# -----------------------------------
st.title("🧩 Wholesale Customer Segmentation")
st.write("Customer grouping based on purchasing behavior using K-Means clustering")

# -----------------------------------
# Load Dataset
# -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Wholesale customers data.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------------
# Feature Selection
# -----------------------------------
st.subheader("Selected Spending Features")

spending_features = [
    "Fresh", "Milk", "Grocery",
    "Frozen", "Detergents_Paper", "Delicassen"
]

df_spending = df[spending_features]
st.write(spending_features)

# -----------------------------------
# Data Scaling
# -----------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_spending)

st.success("Data scaled successfully for distance-based clustering")

# -----------------------------------
# Choose Number of Clusters
# -----------------------------------
st.subheader("Choose Number of Clusters (K)")

k = st.slider("Select K", min_value=2, max_value=8, value=3)

# -----------------------------------
# K-Means Clustering
# -----------------------------------
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

st.success(f"Clustering completed with K = {k}")

# -----------------------------------
# Cluster Visualization
# -----------------------------------
st.subheader("Cluster Visualization")

x_feature = st.selectbox("X-axis feature", spending_features, index=2)
y_feature = st.selectbox("Y-axis feature", spending_features, index=4)

plt.figure(figsize=(7,5))
sns.scatterplot(
    x=df[x_feature],
    y=df[y_feature],
    hue=df["Cluster"],
    palette="viridis"
)

# Plot cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centers[:, spending_features.index(x_feature)],
    centers[:, spending_features.index(y_feature)],
    c="red",
    s=200,
    marker="X",
    label="Centers"
)

plt.xlabel(x_feature)
plt.ylabel(y_feature)
plt.title("Customer Clusters")
plt.legend()
st.pyplot(plt)

# -----------------------------------
# Cluster Profiling
# -----------------------------------
st.subheader("Cluster Profiling (Average Spending)")

cluster_profile = df.groupby("Cluster")[spending_features].mean().round(2)
st.dataframe(cluster_profile)

# -----------------------------------
# Business Insights
# -----------------------------------
st.subheader("Business Insights")

for cluster in cluster_profile.index:
    st.markdown(f"### Cluster {cluster}")
    st.write(cluster_profile.loc[cluster])

    st.markdown("**Suggested Strategy:**")
    if cluster_profile.loc[cluster].max() == cluster_profile.loc[cluster]["Grocery"]:
        st.write("• Focus on retail clients with bulk grocery promotions")
    elif cluster_profile.loc[cluster].max() == cluster_profile.loc[cluster]["Fresh"]:
        st.write("• Target hotels/restaurants with fresh inventory planning")
    else:
        st.write("• Offer personalized discounts and upselling opportunities")

# -----------------------------------
# Stability Check
# -----------------------------------
st.subheader("Stability Check")

kmeans_alt = KMeans(n_clusters=k, random_state=99)
clusters_alt = kmeans_alt.fit_predict(X_scaled)

st.write(
    "Number of customers with same cluster assignment:",
    np.sum(clusters == clusters_alt),
    "out of",
    len(clusters)
)

# -----------------------------------
# Limitation
# -----------------------------------
st.subheader("Model Limitation")

st.markdown("""
- K-Means assumes spherical clusters  
- Sensitive to feature scaling  
- Requires pre-defined number of clusters (K)  
""")
