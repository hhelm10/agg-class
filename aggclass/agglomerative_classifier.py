import numpy as np

from sklearn.cluster import AgglomerativeClustering 
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import pairwise_distances

from scipy.special import softmax

try:
    from genieclust import Genie
except:
    pass

from joblib import delayed, Parallel
from .helpers import get_tree_distances, get_decision_paths, stratified_sample, gem


class AgglomerativeClassifier:
    def __init__(self, n_clusters=2, affinity='cosine', linkage='average',
                n_neighbors=1,
                 max_tree_distance=None,
                 soft_max_multiplier=1,
                 gem_p=1,
                classes=None):
        
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.linkage = linkage
        
        self.n_neighbors = n_neighbors
                
        self.classes_ = classes
        
        self.max_tree_distance=max_tree_distance
        self.soft_max_multiplier=soft_max_multiplier
        
        self.gem_p = gem_p
        
        
    def fit(self, X, y):
        self.n, d = X.shape

        self.labeled_inds = np.where(y != -1)[0]
                
        if self.classes_ is None:
            self.classes_ = np.unique(y[self.labeled_inds])
            
        if -1 in self.classes_:
            self.classes_ = self.classes_[1:]
        
        if self.linkage=='gini' or self.linkage=='bonferroni':
            self.model = Genie(n_clusters=self.n_clusters, 
                compute_all_cuts=False,
                affinity=self.affinity, 
                exact=True,
                compute_full_tree=True)

            self.model.fit(X)
        else:  
            if self.linkage != 'ward':
                pairwise_distances_ = pairwise_distances(X, metric=self.affinity, n_jobs=1)

                self.model = AgglomerativeClustering(n_clusters=self.n_clusters,
                    affinity='precomputed', 
                    linkage=self.linkage,
                    compute_full_tree=True,
                    ).fit(pairwise_distances_)

                
                del pairwise_distances_
            else:
                self.model = AgglomerativeClustering(n_clusters=self.n_clusters, 
                    affinity=self.affinity, 
                    linkage=self.linkage,
                    compute_full_tree=True
                    )
                self.model.fit(X)
                    
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.affinity)
        self.nn.fit(X)

        
        labeled_inds_by_class = [np.where(y[self.labeled_inds] == c)[0] for c in self.classes_]
        decision_paths, counts = get_decision_paths(self.n, self.model.children_)
        
        self._get_tree_distances(decision_paths, counts)        
        self._get_similarities_to_classes(labeled_inds_by_class)

        return self
        
                                    
    def _get_tree_distances(self, decision_paths, counts):
        self.tree_distances = get_tree_distances(self.n, decision_paths, self.labeled_inds, counts, self.max_tree_distance)
        self.scores = np.log(self.tree_distances + 1) + 1
        self.scores = 1 / self.scores
        self.scores = softmax(self.soft_max_multiplier * self.scores, axis=1)
        
        
    def _get_similarities_to_classes(self, labeled_inds_by_class):
        self.similarities_to_classes = np.zeros((self.n, len(self.classes_)))
        
        for i, ibc in enumerate(labeled_inds_by_class):
            self.similarities_to_classes[:, i] = gem(self.scores[:, ibc].T, p=self.gem_p)
                                              
        self.similarities_to_classes = softmax(self.similarities_to_classes, axis=1)
        
        
    def predict_proba(self, X):
        _, neighbor_inds = self.nn.kneighbors(X)
        scores = np.mean(self.similarities_to_classes[neighbor_inds], axis=1)
        return softmax(scores, axis=1)
    
    
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class ProjectionAgglomerativeClassifier(AgglomerativeClassifier):
    def __init__(self, 
                 projector=None, projection_kwargs={},
                 n_clusters=2, affinity='cosine', linkage='average',
                 n_neighbors=1,
                 max_tree_distance=None,
                 gem_p=1,
                 soft_max_multiplier=1,
                 exact=False,
                classes=None):
        
        if projector == "gaussian":
            self.projector=GaussianRandomProjection
        else:
            self.projector=projector

        self.projection_kwargs=projection_kwargs
        
        super().__init__(n_clusters = n_clusters, affinity = affinity, linkage = linkage,
                         n_neighbors = n_neighbors,
                         classes = classes, max_tree_distance=max_tree_distance,
                         gem_p=gem_p,
                         soft_max_multiplier=soft_max_multiplier)
        
        
    def fit(self, X, y):
        if self.projector is None:
            super().fit(X, y)
        else:
            self.projector = self.projector(**self.projection_kwargs)
            self.projector.fit(X)
            super().fit(self.projector.transform(X), y)
  

    def predict_proba(self, X):
        if self.projector is None:
            return super().predict_proba(X)
        else:
            return super().predict_proba(self.projector.transform(X))

    
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class AgglomerativeEnsemble:
    def __init__(self, n_estimators=50, 
                 p_inbag=0.67, 
                 projector=None, projection_kwargs={},
                 classes=None,
                 affinity='cosine', linkage='average',
                 gem_p=1,
                 n_neighbors=1,
                 max_tree_distance=None,
                 n_jobs=1):

        self.n_estimators=n_estimators
        self.p_inbag=p_inbag
        
        if projector == "gaussian":
            self.projector=GaussianRandomProjection
        else:
            self.projector=projector

        self.projection_kwargs=projection_kwargs
        
        self.classes_=classes

            
        self.affinity=affinity
        self.linkage=linkage
        self.n_neighbors=n_neighbors
        self.max_tree_distance=max_tree_distance

        self.gem_p = gem_p

        self.n_jobs=n_jobs
        self.ensemble = []
    
        
    def fit(self, X, y):
        self.unlabeled_inds = np.where(y == -1)[0].astype(int)
        self.labeled_inds = np.where(y != -1)[0].astype(int)
        
        if self.classes_ is None:
            self.classes_ = np.unique(y)

        if -1 in self.classes_:
            self.classes_ = self.classes_[1:]
                             
        condensed_func = lambda x: self._train_agg_class(X, y, stratified=True)
        func_tuples = np.zeros(self.n_estimators)
                
        self.ensemble = Parallel(n_jobs=self.n_jobs)(delayed(condensed_func)(tuple_) for tuple_ in func_tuples)
            

    def _train_agg_class(self, X, y, stratified=True):                
        if len(self.labeled_inds) == len(y):
            all_supervised=True
        else:
            all_supervised=False

        if self.p_inbag >= 1:
            bag_inds = np.arange(X.shape[0])
        else:
            replace=True
            sbag_inds = stratified_sample(self.labeled_inds, p=self.p_inbag, replace=replace)
            if all_supervised:
                bag_inds = sbag_inds
            else:
                ssbag_inds = np.random.choice(self.unlabeled_inds, size=int(len(self.unlabeled_inds)*self.p_inbag), replace=replace)
                bag_inds = np.concatenate((self.labeled_inds[sbag_inds], ssbag_inds))
        

        agg_class = ProjectionAgglomerativeClassifier(projector=self.projector, projection_kwargs=self.projection_kwargs, 
            affinity=self.affinity, linkage=self.linkage,
            n_neighbors=self.n_neighbors,
            max_tree_distance=self.max_tree_distance,
            gem_p=self.gem_p,
            classes=self.classes_)


        agg_class.fit(X[bag_inds], y[bag_inds])
                
        return agg_class
    

    def predict_proba(self, X):
        condensed_predict = lambda agg_class: agg_class.predict_proba(X)
        return np.mean(Parallel(n_jobs=self.n_jobs)(delayed(condensed_predict)(agg_class) for agg_class in self.ensemble), axis=0)
        

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)