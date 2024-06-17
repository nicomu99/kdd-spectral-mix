import numpy as np
import scipy

class DataLoader():
    def load_dataset(self, dataset_name):
        adjacency_matrix = None
        attribute_matrix = None
        true_labels = None
        if dataset_name not in ['acm', 'dblp', 'flickr', 'imdb']:
            print("Invalid dataset name")

        if dataset_name == 'acm':
            adjacency_matrix, true_labels = self.construct_matrix('acm', ['PAP', 'PLP'], construct_adjacency=False)   
            attribute_matrix, _ = self.construct_matrix('acm', ['feature'], construct_adjacency=False)
        elif dataset_name == 'dblp':
            adjacency_matrix, true_labels = self.construct_matrix('dblp', ['APNet', 'citation', 'co_citation', 'coauthorNet'], 8401)
        elif dataset_name == 'flickr':
            adjacency_matrix, true_labels =  self.construct_matrix('flickr', ['layer0', 'layer1'])
        else:
            adjacency_matrix, attribute_matrix, true_labels = self.construct_matrix('imdb', [])
            attribute_matrix = attribute_matrix.astype(int)

        return {
            'adjacency_matrix': adjacency_matrix,
            'attribute_matrix': attribute_matrix,
            'true_labels': true_labels
        }


    def construct_adjacency_matrix(self, edge_list, graph_size):
        max_index = np.max(edge_list)
        n = max_index + 1
        if graph_size > n:
            n = graph_size

        adjacency_matrix = np.zeros((n, n), dtype=np.int32)
        for edge in edge_list:
            weight = 1
            if len(edge) > 2:
                weight = edge[2]
            adjacency_matrix[edge[0], edge[1]] = weight
            adjacency_matrix[edge[1], edge[0]] = weight

        return adjacency_matrix

    def read_matrix(self, path):
        matrix = scipy.io.loadmat(path)
        for key in matrix.keys():
            if not key.startswith('__'):
                return matrix[key].astype(int)
        raise TypeError('Matrix not found!')

    def construct_matrix(self, dataset, file_names=[], graph_size=0, construct_adjacency=True):
        true_labels = np.loadtxt(f'data/{dataset}/ground_truth.txt')
        if dataset == 'imdb':
            imdb = scipy.io.loadmat('data/imdb/imdb.mat')
            return np.stack((imdb['MAM'], imdb['MDM']), axis=-1), imdb['feature'], true_labels
        loaded_matrices = []
        for file in file_names:
            layer = self.read_matrix(f'data/{dataset}/{file}.mat')
            if construct_adjacency:
                loaded_matrices.append(self.construct_adjacency_matrix(layer, graph_size))
            else:
                loaded_matrices.append(layer)

        if len(loaded_matrices) > 1:
            return np.stack(loaded_matrices, axis=-1), true_labels
        return np.array(loaded_matrices[0]), true_labels