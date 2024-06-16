import numpy as np
from sklearn.cluster import KMeans

class SpectralMix:
    def __init__(self, d=7, k=7, iter=100, etol=1e-8):
        self.iter = iter
        self.etol = etol
        self.d = d
        self.k = k

    def fit(self, adjacency_matrix, attribute_matrix):
        self.adjacency_matrix = adjacency_matrix
        self.attribute_matrix = attribute_matrix
        self.num_nodes, _, self.num_rels = adjacency_matrix.shape               # number of nodes and relation types

        self.num_attr = self.attribute_matrix.shape[1]                          # number of attribute categories
        self.count_attr = []                                                    # holds the count of each category in each attribute
        for a in range(self.num_attr):
            values, counts = np.unique(self.attribute_matrix[:, a], return_counts=True)
            if values[0] == -1:
                counts = counts[1:]
            self.count_attr.append(counts)

        self.o = np.random.randn(size=(self.num_nodes, self.d))
        self.m = np.random.randn(size=(self.num_attr, self.d))
        
        self.weighting_factor = np.zeros(self.num_rels + self.num_attr)
        self.sum_weight = np.zeros(self.num_rels + self.num_attr)

        # Iterate over all edges in each relation type and find the relation type with highest total weight
        self.max_weight = 0
        self.max_index = -1
        for r in range(self.num_rels):
            rel_edges = self.adjacency_matrix[:, :, r]
            self.sum_weight[r] = np.sum(rel_edges)
            if self.sum_weight[r] > self.max_weight:
                self.max_weight = self.sum_weight
                self.max_index = r

        # iterate over the number of attributes and the number of distinct categories in each attribute
        # take the count of each category as the weight of this category
        for a in range(self.num_attr):
            self.sum_weight[self.num_rels + a] += sum(self.count_attr[a])
            if self.sum_weight[self.num_rels + a] > self.max_weight:
                self.max_weight = self.sum_weight
                self.max_index = self.num_rels + a

        
        self.weighting_factor = self.sum_weight / self.max_weight

        self.sum_g = np.zeros(size=(self.num_nodes,))
        for r in range(self.num_rels):
            non_zero = np.count_nonzero(self.adjacency_matrix[:, :, r], axis=1)
            self.sum_g += non_zero.reshape(-1, 1) * self.weighting_factor[r]

        # add the weighting of the attributes to the
        for a in range(self.num_attr): 
            contains_negative_attr = np.any(self.attribute_matrix[:, a] < 0, axis=1)
            contains_negative_attr = (~contains_negative_attr).astype(int)
            contains_negative_attr = contains_negative_attr.reshape(-1, 1)
            self.sum_g += contains_negative_attr * self.weighting_factor[self.num_rels + a]

        iterations = 0
        while iterations < self.iter:
            for r in range(self.num_rels):
                for p in range(self.num_nodes):
                    for l in range(self.d): 
                        self.o[:, l] += self.o[p, l] * self.weighting_factor[self.num_rels] * self.adjacency_matrix[:, p, r] / self.sum_g[:]

            # attribute contribution
            mask = self.attribute_matrix != -1
            attribute_weighting_factors = self.weighting_factor[self.num_rels:self.num_rels + self.num_attr]
            scaled_weighting_factors = attribute_weighting_factors / self.sum_g[:, np.newaxis]
            scaled_weighted_attributes = mask * scaled_weighting_factors
            for j in range(self.num_attr):
                if np.any(mask[:, j]): 
                    for l in range(self.d):
                        self.o[:, l] += scaled_weighted_attributes[:, j] * self.m[j, l]
            self.o, _ = np.linalg.qr(self.o)
            for l in range(self.d):
                for j in range(self.num_attr):
                    pos_boolean = self.attribute_matrix[:, j] != -1
                    pos_indices = self.attribute_matrix[pos_boolean, j]
                    self.m[j, l] = np.sum(self.o[pos_boolean, l] / self.count_attr[j, pos_indices])

            iterations += 1

        kmeans_model = KMeans(n_clusters=self.k).fit(self.o)
        self.labels = kmeans_model.labels_

        return self