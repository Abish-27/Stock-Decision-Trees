"""CSC111 Project 2: Smart Trades- Trees

Instructions (READ THIS FIRST!)
===============================

This Python module contains code to implement a decision tree classifier for predicting stock trades.
It includes classes for tree nodes and decision tree construction, using entropy and information gain to determine optimal splits.

Copyright and Usage Information
===============================

This file is provided solely for the professional use of CSC111 adminstrators
at the University of Toronto St. George campus. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited. For more information on copyright for CSC111 materials,
please consult CSC111 Course Syllabus.

This file is Copyright (c) Smart Trades Team- Shaurya Sareen, Abish Kulkarni, Irin Jin
"""

from typing import Optional, Any
import numpy as np


class TreeNode:

    """
    Represents a node in the decision tree.

    Instance Attributes:
        - feature: Index of feature used for splitting.
        - threshold: Threshold value for splitting.
        - left: Left subtree (values <= threshold).
        - right: Right subtree (values > threshold).
        - prediction: If leaf node, store the predicted class.

    Representation Invariants:
        - not(self.prediction is not None) or(self.feature is None and self.threshold is None and self.left is None and self.right is None)
        - not(self.feature is not None and self.threshold is not None) or (self.left is not None and self.right is not None)
    """

    feature: Optional[int]
    threshold: Optional[float]
    left: Optional['TreeNode']
    right: Optional['TreeNode']
    prediction: Optional[int]

    def __init__(self, feature: Optional[int] = None, threshold: Optional[float] = None, left: Optional['TreeNode'] = None,
                 right: Optional['TreeNode'] = None,
                 prediction: Optional[int] = None) -> None:
        """
        Sets the initial left and right subtrees, index of the feature, threshold value for splitting, and prediction.
        """

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction


class DecisionTree:
    """
    A Decision Tree classifier that uses entropy as the splitting criterion.
    The decision tree is built recursively by selecting the feature that provides the highest
    information gain at each node, up to the specified maximum depth or when the tree achieves pure node classification.

    Instance Attributes:
        - max_depth: Maximum depth of the tree.
        - root: Root node of the trained decision tree.

    Representation Invariants:
        - if self.root is not None, it must be a valid TreeNode
        - if self.max_depth is not None, it must be a positive integer
    """

    max_depth: Optional[int]
    root: Optional[TreeNode]

    def __init__(self, max_depth: Optional[int] = None) -> None:
        """Initializes the DecisionTree with an initial max depth and no root."""
        self.max_depth = max_depth
        self.root = None

    def best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[Any, Any]:
        """
        Identifies the best feature and threshold to split on by maximizing information gain.

        >>> tree = DecisionTree()
        >>> x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> y = np.array([0, 0, 1, 1])
        >>> tree.best_split(x, y) in [(0, 3), (0, 5), (1, 4), (1, 6)]
        True
        """
        best_gain = 0
        best_feature = None
        best_threshold = None

        for feature_idx in range(x.shape[1]):
            feature_column = np.array([row[feature_idx] for row in x])
            thresholds = np.unique(feature_column)
            for threshold in thresholds:
                gain = self.information_gain(feature_column, y, threshold)
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature_idx, threshold

        return best_feature, best_threshold

    def information_gain(self, x_column: np.ndarray, y: np.ndarray, threshold: float) -> float:
        """
        Computes the information gain for a given threshold.
        Information gain measures how much splitting on a feature reduces entropy.

        >>> tree = DecisionTree()
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([0, 0, 1, 1, 1])
        >>> float(round(tree.information_gain(x, y, 3), 4))
        0.42
        """
        left_mask = x_column <= threshold
        right_mask = x_column > threshold

        left_entropy = self.entropy(y[left_mask])
        right_entropy = self.entropy(y[right_mask])

        n = len(y)
        left_weight = sum(left_mask) / n
        right_weight = sum(right_mask) / n

        return self.entropy(y) - (left_weight * left_entropy + right_weight * right_entropy)

    def entropy(self, y: np.ndarray) -> float:
        """
        Entropy is a measure of impurity in a dataset.
        A lower entropy value indicates that the dataset is purer (i.e., mostly one class),
        whereas a higher entropy value suggests more class diversity.

        >>> tree = DecisionTree()
        >>> float(tree.entropy(np.array([0, 0, 1, 1, 1])))
        0.9709505944546686
        """
        counts = np.unique(y, return_counts=True)[1]
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities))

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Initializes the root of the decision tree by calling `build_tree`,
        which recursively partitions the data until stopping conditions are met (e.g., maximum depth
        reached or all samples in a node belong to the same class).
        """
        self.root = self.build_tree(x, y)

    def build_tree(self, x: np.ndarray, y: np.ndarray, depth: int = 0) -> TreeNode:
        """
        Recursively builds the decision tree using the best feature splits.
        """

        if depth == self.max_depth or len(np.unique(y)) == 1:
            return TreeNode(prediction=np.bincount(y).argmax())

        best_feature, best_threshold = self.best_split(x, y)
        if best_feature is None:
            return TreeNode(prediction=np.bincount(y).argmax())

        left_mask = np.array([element[best_feature] <= best_threshold for element in x])
        right_mask = np.array([element[best_feature] > best_threshold for element in x])

        left_subtree = self.build_tree(x[left_mask], y[left_mask], depth + 1)
        right_subtree = self.build_tree(x[right_mask], y[right_mask], depth + 1)

        return TreeNode(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for multiple samples by traversing the trained decision tree.

        This function applies `predict_one` to each sample in the input dataset.
        """
        return np.array([self.predict_one(element, self.root) for element in x])

    def predict_one(self, x: np.ndarray, node: TreeNode) -> int:
        """
        Predicts the class label for a single sample by traversing the decision tree.

        The function starts at the root and follows the tree structure by comparing the sample's
        feature values against node thresholds until a leaf node is reached.
        """
        if node.prediction is not None:
            return int(node.prediction)
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['numpy', 'typing'],
        'allowed-io': [],
        'max-line-length': 140,
        'disable': ['R0913']
    })
