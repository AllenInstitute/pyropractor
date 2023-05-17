import numpy as np
import tensorflow as tf
from tensorflow_graphics.nn.loss import chamfer_distance
from tqdm import tqdm


def minimize_chamfer_dist(align_feature_arr, reference_feature_arr, epochs,
                          n_random_starts, learning_rate):
    """
    Will minimize the chamfer distance between a given point cloud and a reference point cloud

    Parameters
    ----------
    align_feature_arr : array-like of shape (n_samples, m)
        where `n_samples` is the number of samples
        and `m` is equal to 1 + the number of static feature(s).
        The feature to be aligned should be the first column
        followed by the static feature(s). The first column
        of this array will be transformed so that the chamfer
        distance between this (n_samples, m) point cloud and
        the reference_feature_arr point cloud is minimized.

    reference_feature_arr : array-like of shape (p_samples, m)
        where `p_samples` is the number of samples
        and `m` is equal to 1 + the number of static features.
        The feature being aligned should be in the first column
        followed by the static features.

    epochs : int, number of iterations to run while trying
        to minimize chamfer distance

    n_random_starts : int, number of times to run through
        'epochs' number of iterations with random weight
        and bias initialization. If n_random_starts = 5
        and epochs = 1000, the weight and bias variables
        will be initialized randomly 5 times and in each
        initialization 1000 epochs will be run and the
        weight & bias that provide minimum chamfer distance
        will be returned. This is to help with local
        minimum in gradient descent.

    learning_rate : float, learning rate for gradient descent

    Returns
    ----------
    results : dict, dictionary with the following key/values
        weight : optimal weight
        bias : optimal bias
        original_chamfer_distance : the chamfer distance between the unaligned features
        minimized_chamfer_distance : the chamfer distance between the aligned features
        new_aligned_feature_array : the aligned feature array

    """
    # convert input numpy arrays to tensors with consisten dtypes
    align_feature_arr = tf.convert_to_tensor(align_feature_arr, dtype=np.float32)
    reference_feature_arr = tf.convert_to_tensor(reference_feature_arr, dtype=np.float32)

    # baseline chamfer distace
    original_chamf_dist = chamfer_distance.evaluate(align_feature_arr, reference_feature_arr)

    # initialize optimize variables
    best_W = None
    best_b = None
    minimized_chamf_dist = np.inf
    new_align_feature_arr = None
    for _ in range(n_random_starts):

        # randomly initiate weight and bias
        W = tf.Variable(tf.random.normal([1]), trainable=True, name='weight')
        b = tf.Variable(tf.random.normal([1]), trainable=True, name='bias')

        @tf.function
        def cost(W, b, return_trans=False):

            # update feature with weight and bias
            align_new_feature = align_feature_arr[:, 0] * W[0] + b
            align_new_feature = tf.reshape(align_new_feature, (len(align_new_feature), 1))

            if align_feature_arr.shape[1] > 1:
                # if there are static features, create m-dimensional point cloud
                align_feature_arr_statics = tf.slice(align_feature_arr,
                                                     begin=[0, 1],
                                                     size=[align_feature_arr.shape[0], align_feature_arr.shape[1] - 1])
                align_new_feature_arr = tf.concat((align_new_feature, align_feature_arr_statics), axis=1)

            else:
                # otherwise this is a one-dimensional point cloud
                align_new_feature_arr = align_new_feature

            ch_dist = chamfer_distance.evaluate(align_new_feature_arr, reference_feature_arr)
            if not return_trans:
                return ch_dist
            else:
                # return transformed feature
                return ch_dist, align_new_feature_arr

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        trainable_vars = [W, b]

        for _ in tqdm(range(epochs)):
            with tf.GradientTape() as tp:
                cost_fn = cost(W, b)
            gradients = tp.gradient(cost_fn, trainable_vars)
            optimizer.apply_gradients(zip(gradients, trainable_vars))

        this_it_chamf_dist, this_it_align_feature_arr = cost(W, b, True)

        if this_it_chamf_dist < minimized_chamf_dist:
            minimized_chamf_dist = this_it_chamf_dist
            new_align_feature_arr = this_it_align_feature_arr
            best_W = W
            best_b = b

    results = {
        "weight": best_W.numpy()[0],
        "bias": best_b.numpy()[0],
        "original_chamfer_distance": original_chamf_dist.numpy(),
        "minimized_chamfer_distance": minimized_chamf_dist.numpy(),
        "new_aligned_feature_array": new_align_feature_arr[:, 0].numpy()
    }

    return results


class ChamferAlign:
    """
    Will minimize the chamfer distance between feature point clouds.
    This function will iterate over each feature, and minimize
    the chamfer distance between the reference and aligning
    dataset. Static features can be set so that chamfer distance
    is not only minimized in 1-dimension for each feature.

    Parameters
    ----------
    static_feature_index : None, int or list. column index(s) for
        feature(s) that will remain unadjusted, but are used to
        add dimensionality to aligning point-clouds of all
        other features. For example, if static_feature_index = 3,
        all columns (features) except the 3rd will be aligned to the
        reference dataset. During the alignment of each
        of the other features though, the 3rd column will be used to
        turn the 1-dimensional point clouds into  2-dimensional
        point-clouds.

    epochs : int, number of iterations to run while trying
        to minimize chamfer distance. Must be > 1

    n_random_starts : int, number of times to run through
        'epochs' number of iterations with random weight
        and bias initialization. If n_random_starts = 5
        and epochs = 1000, the weight and bias variables
        will be initialized randomly 5 times and in each
        initialization 1000 epochs will be run and the
        weight & bias that provide minimum chamfer distance
        will be returned. This is to help with local
        minimum in gradient descent. Must be > 0

    learning_rate : float, learning rate for gradient descent

    subsample : float, value between 0 and 1 denoting percentage
        of data to subsample in each epoch

    Attributes
    ----------
    transform_data : dict, is populated after self.align is called.
        This stores metadata from the alignment process. Keys
        represent feature column indexes and values are nested
        dictionaries. Note any static features will not appear in
        this dictionary since they are not adjusted. The key/value
        pairs for nested dictionaries are as follows:
            weight : float, the optimized weight for linearly transforming
                     the ith feature
            bias : float, the optimized bias for linearly transforming
                     the ith feature
            original_chamfer_distance : chamfer distance between point clouds
                before alignment
            minimized_chamfer_distance : chamfer distance between point clouds
                after alignment
            new_aligned_feature_array : the aligned version of the ith feature

    # Examples
    # --------
    >>>>import numpy as np
    >>>>import matplotlib.pyplot as plt
    >>>>from pyropractor.chamfer import ChamferAlign

    >>>>reference_feat = np.random.normal(0,0.5,100)
    >>>>reference_feat[0:30] = abs(reference_feat[0:30])*10
    >>>>reference_feat[80:] = abs(reference_feat[80:])*20
    >>>>misaligned_feat = (reference_feat*0.3)+0.113
    >>>>ys = np.arange(0,1,1/len(reference_feat))

    >>>>reference_X = np.stack((reference_feat,ys)).T
    >>>>misaligned_X = np.stack((misaligned_feat,ys)).T

    >>>>chmf = ChamferAlign(epochs=500, subsample=1.0, n_random_starts=5,static_feature_index=1,learning_rate=0.01)
    >>>>aligned_X = chmf.align(align_X=misaligned_X, reference_X=reference_X)

    >>>>plt.scatter(reference_feat,ys,label='reference',c='k',alpha=0.75)
    >>>>plt.scatter(misaligned_feat,ys,label='misaligned',c='firebrick',alpha=0.75)
    >>>>plt.scatter(aligned_X[:,0],ys,label='aligned',c='steelblue',alpha=0.75)
    >>>>plt.xlabel("Tunable Feature")
    >>>>plt.ylabel("Static Feature")
    >>>>plt.title("Transforming Misaligned Data to A Reference PointCloud")
    >>>>plt.legend()
    """

    def __init__(
            self,
            static_feature_index=[0],
            epochs=1000,
            n_random_starts=5,
            learning_rate=0.001,
            subsample=1.0,
    ):
        self.epochs = epochs
        self.subsample = subsample
        self.n_random_starts = n_random_starts
        self.static_feature_index = static_feature_index
        self.learning_rate = learning_rate
        self.transform_data = {}

    def _validate_params(self, reference_X, align_X):
        accepted_param_types = {
            "epochs": [[int], [lambda x: x > 1]],
            "subsample": [[float], [lambda x: 0 < x <= 1]],
            "n_random_starts": [[int], [lambda x: x > 0]],
            "static_feature_index": [[type(None), int, list, set], [lambda x: True]],
            "learning_rate": [[float], [lambda x: True]]
        }

        for attr_name, qc_list in accepted_param_types.items():
            accept_list = qc_list[0]
            qc_lambda = qc_list[1][0]

            if not qc_lambda(self.__getattribute__(attr_name)):
                raise ValueError(
                    f"Invalid value passed for {attr_name}. Check docstrings for requirements"
                )

            assigned_atr_type = type(self.__getattribute__(attr_name))
            if assigned_atr_type not in accept_list:
                raise ValueError(
                    f"Unexpected type passed for {attr_name} ({assigned_atr_type}) "
                    f"accepted input types for this attribute are: {accept_list}"
                )

        ref_n_feats = reference_X.shape[1]
        align_n_feats = align_X.shape[1]
        if ref_n_feats != align_n_feats:
            raise ValueError(
                f"reference_X has {ref_n_feats} features, but align_X has {align_n_feats}"
                f"expecting same number of features per array."
            )

        feature_indexes = list(range(0, ref_n_feats))
        if self.static_feature_index is not None:
            if isinstance(self.static_feature_index, int):
                if self.static_feature_index not in feature_indexes:
                    raise ValueError(
                        f"static_feature_index ({self.static_feature_index}) is out of range for the provided feature "
                        f"matrices."
                        f"Expected None or at least one value in the range between 0 and {ref_n_feats}"
                    )
            else:
                if not all([i in feature_indexes for i in self.static_feature_index]):
                    raise ValueError(
                        f"At least one value in static_feature_index ({self.static_feature_index}) is out of range for the provided"
                        f"feature matrices. Values should be in the range between 0 and {ref_n_feats}"
                    )

    def _align(self, reference_X, align_X):
        "Run alignment between two feature matrices"

        aligned_X = np.zeros(align_X.shape)

        n_feats = reference_X.shape[1]
        feature_indexes = list(range(0, n_feats))

        # find features that will be aligned
        static_reference = None
        static_align = None
        if self.static_feature_index is not None:

            aligned_X[:, self.static_feature_index] = align_X[:, self.static_feature_index]
            n_statics = 1
            if isinstance(self.static_feature_index, int):
                feature_indexes.remove(self.static_feature_index)

            else:
                feature_indexes = [i for i in feature_indexes if i not in self.static_feature_index]
                n_statics = len(self.static_feature_index)

            static_reference = reference_X[:, self.static_feature_index].reshape(-1, n_statics)
            static_align = align_X[:, self.static_feature_index].reshape(-1, n_statics)

        for idx in feature_indexes:
            print(f"Aligning feature at column index {idx}")
            reference_feature = reference_X[:, idx].reshape(-1, 1)
            align_feature = align_X[:, idx].reshape(-1, 1)
            if self.static_feature_index is not None:
                reference_feature = np.hstack((reference_feature, static_reference))
                align_feature = np.hstack((align_feature, static_align))

            results_dict = minimize_chamfer_dist(
                align_feature_arr=align_feature,
                reference_feature_arr=reference_feature,
                epochs=self.epochs,
                n_random_starts=self.n_random_starts,
                learning_rate=self.learning_rate,
            )
            self.transform_data[idx] = results_dict

            aligned_X[:, idx] = results_dict['new_aligned_feature_array']

        return aligned_X

    def align(self, reference_X, align_X):
        """Align the features in align_X to reference_X

        Parameters
        ----------
        reference_X : array-like of shape (n_samples, n_features)
            Source data, where `n_samples` is the number of samples
            and `n_features` is the number of features. This is the
            reference data that will be aligned to.

        align_X : array-like of shape (p_samples, n_features)
            Source data, where `p_samples` is the number of samples
            and `n_features` is the number of features. n_samples and
            m_samples need not be equivalent. This is the data that
            will be aligned to the reference data.


        Returns
        -------
        aligned_X : array-like of shape (p_samples, n_features)
            This is the data that has been transformed to align
            with the reference data
        """
        self._validate_params(reference_X, align_X)
        aligned_X = self._align(reference_X, align_X)
        return aligned_X
        # return self
