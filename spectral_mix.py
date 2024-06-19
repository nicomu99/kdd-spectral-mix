import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

class SpectralMix:
    def __init__(self, d=7, n_clusters=7, iter=50, etol=1e-8):
        self.iter = iter
        self.etol = etol
        self.d = d
        self.k = n_clusters
        self.labels_ = None

    def fit(self, adjacency_matrix, attribute_matrix):

        self.adjacency_matrix = adjacency_matrix
        self.attribute_matrix = attribute_matrix
        self.num_nodes, _, self.num_rels = adjacency_matrix.shape               # number of nodes and relation types

        if attribute_matrix is not None:
            self.num_attr = self.attribute_matrix.shape[1]                      # number of attribute categories
        else:
            self.num_attr = 0

        self.mask = self.attribute_matrix != -1
        
        self.count_attr = []                                                    # holds the count of each category in each attribute
        self.num_cat = 0
        for a in range(self.num_attr):
            values, counts = np.unique(self.attribute_matrix[:, a], return_counts=True)
            if values[0] == -1:
                counts = counts[1:]
            self.count_attr.append(counts)
            self.num_cat += len(counts)

        self.o = np.random.randn(self.num_nodes, self.d)
        self.m = np.random.randn(self.num_cat, self.d)
        
        self.weighting_factor = np.zeros(self.num_rels + self.num_attr)
        self.sum_weight = np.zeros(self.num_rels + self.num_attr)

        # Iterate over all edges in each relation type and find the relation type with highest total weight
        self.max_weight = 0
        self.max_index = -1
        for r in range(self.num_rels):
            rel_edges = self.adjacency_matrix[:, :, r]
            self.sum_weight[r] = np.sum(rel_edges)
            if self.sum_weight[r] > self.max_weight:
                self.max_weight = self.sum_weight[r]
                self.max_index = r

        # iterate over the number of attributes and the number of distinct categories in each attribute
        # take the count of each category as the weight of this category
        for a in range(self.num_attr):
            self.sum_weight[self.num_rels + a] += sum(self.count_attr[a])
            if self.sum_weight[self.num_rels + a] > self.max_weight:
                self.max_weight = self.sum_weight
                self.max_index = self.num_rels + a

        self.weighting_factor = self.sum_weight / self.max_weight

        self.sum_g = np.zeros(shape=(self.num_nodes,))
        for r in range(self.num_rels):
            non_zero = np.count_nonzero(self.adjacency_matrix[:, :, r], axis=1)
            self.sum_g += non_zero * self.weighting_factor[r]

        # Add the attribute contribution to each node
        # The attribute contribution only gets added if the attribute is not -1 for a node 
        if self.num_attr > 0:
            attribute_weights = self.weighting_factor[self.num_rels : self.num_rels + self.num_attr]
            weighted_contributions = self.mask * attribute_weights  # Shape: (num_nodes, num_attr)
            self.sum_g += np.sum(weighted_contributions, axis=1)
        
        identity_mask = np.eye(self.num_nodes, dtype=bool)
        # Prepare the weighting factor array and broadcasting
        weighting_factor_matrix = self.weighting_factor[self.num_rels : self.num_rels + self.num_attr]  # Shape: (num_attr,)
        weighting_factor_matrix = weighting_factor_matrix / self.sum_g[:, np.newaxis]  # Shape: (num_nodes, num_attr)
        for _ in tqdm(range(self.iter)):
            for r in range(self.num_rels):
                neighbors = self.adjacency_matrix[:, :, r] 
                neighbors = np.where(identity_mask, 0, neighbors)
                for l in range(self.d):
                    o_col = self.o[:, l]
                    weighted_contributions = (self.weighting_factor[r] * neighbors * o_col[np.newaxis, :])
                    contribution_sums = np.sum(weighted_contributions, axis=1)
                    self.o[:, l] += contribution_sums / self.sum_g

            weighted_sum = np.zeros((self.num_nodes, self.d))
            for j in range(self.num_attr):
                weight = weighting_factor_matrix[:, j]
                for l in range(self.d):
                    weighted_sum[:, l] += np.where(self.mask[:, j], weight, 0)

            # Update self.o using the calculated weighted_sum
            self.o += weighted_sum

            self.o, _ = np.linalg.qr(self.o)

            for l in range(self.d):
                for j in range(self.num_attr):
                    for attr_index in range(len(self.count_attr[j])):
                        # Create a boolean mask where attribute_matrix[i, j] != -1
                        mask_j = self.mask[:, j]
                        filtered_o = self.o[mask_j, l]
                        sum_filtered_o = np.sum(filtered_o)
                        self.m[j + attr_index, l] += sum_filtered_o / self.count_attr[j][attr_index]

        return self
    
    def fit_predict(self, adjacency_matrix, attribute_matrix):
        self.fit(adjacency_matrix, attribute_matrix)
        return self.predict()
    
    def predict(self):
        kmeans_model = KMeans(n_clusters=self.k).fit(self.o)
        self.labels_ = kmeans_model.labels_
    
        return self.labels_
