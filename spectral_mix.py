import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

class SpectralMix:
    def __init__(self, adjacency_matrix, attribute_matrix, d=7, k=7, iter=50, etol=1e-8):
        self.iter = iter
        self.etol = etol
        self.d = d
        self.k = k
        self.labels_ = None

        self.adjacency_matrix = adjacency_matrix
        self.attribute_matrix = attribute_matrix
        self.num_nodes, _, self.num_rels = adjacency_matrix.shape               # number of nodes and relation types

        if attribute_matrix is not None:
            self.num_attr = self.attribute_matrix.shape[1]                          # number of attribute categories
        else:
            self.num_attr = 0
        
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

        # add the weighting of the attributes to the
        for a in range(self.num_attr): 
            contains_negative_attr = np.any(self.attribute_matrix[:, a] < 0, axis=0)
            contains_negative_attr = (~contains_negative_attr).astype(int)
            self.sum_g += contains_negative_attr * self.weighting_factor[self.num_rels + a]


    def fit(self, run_clustering=True):
        
        identity_mask = np.eye(self.num_nodes, dtype=bool)
        # attribute contribution
        mask = self.attribute_matrix != -1
        attribute_weighting_factors = self.weighting_factor[self.num_rels:self.num_rels + self.num_attr]
        scaled_weighting_factors = attribute_weighting_factors / self.sum_g[:, np.newaxis]
        scaled_weighted_attributes = mask * scaled_weighting_factors
        for _ in tqdm(range(self.iter)):
            for r in range(self.num_rels):
                neighbors = self.adjacency_matrix[:, :, r] 
                neighbors = np.where(identity_mask, 0, neighbors)
                for l in range(self.d):
                    o_col = self.o[:, l]
                    weighted_contributions = (self.weighting_factor[r] * neighbors * o_col[np.newaxis, :])
                    contribution_sums = np.sum(weighted_contributions, axis=1)
                    self.o[:, l] += contribution_sums / self.sum_g

            for j in range(self.num_attr):
                if np.any(mask[:, j]): 
                    for l in range(self.d):
                        self.o[:, l] += scaled_weighted_attributes[:, j] * self.m[j, l]
            self.o, _ = np.linalg.qr(self.o)
            

            #for i in range(self.num_nodes):
            #    for l in range(self.d):
            #        for j in range(self.num_attr):
            #            for attr_index in range(len(self.count_attr[j])):
            #                if self.attribute_matrix[i, j] != -1:
            #                    self.m[j + attr_index, l] += self.o[i, l] / self.count_attr[j][attr_index]

            # Sum over the 'i' dimension (axis 0) of self.o
            o_sum = np.sum(self.o, axis=0)

            for l in range(self.d):
                for j in range(self.num_attr):
                    for attr_index in range(len(self.count_attr[j])):
                        # Create a boolean mask where attribute_matrix[i, j] != -1
                        mask_j = mask[:, j]
                        filtered_o = self.o[mask_j, l]
                        sum_filtered_o = np.sum(filtered_o)
                        self.m[j + attr_index, l] += sum_filtered_o / self.count_attr[j][attr_index]



        if run_clustering:
            kmeans_model = KMeans(n_clusters=self.k).fit(self.o)
            self.labels_ = kmeans_model.labels_

        return self
    
    def predict(self):
        kmeans_model = KMeans(n_clusters=self.k).fit(self.o)
        self.labels_ = kmeans_model.labels_
    
        return self.labels_
