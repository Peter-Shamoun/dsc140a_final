import numpy as np
import time
import random
from collections import Counter
import inspect

def train_test_split(words, labels, test_size=0.2, random_state=None):
    """
    Split data into training and test sets with stratification.
    
    Parameters:
    -----------
    words : list of str
        The list of words
    labels : list of str
        The list of labels
    test_size : float
        The proportion of data to use for testing
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (train_words, test_words, train_labels, test_labels)
    """
    if random_state is not None:
        random.seed(random_state)
    
    # Create indices for each class
    spanish_indices = [i for i, label in enumerate(labels) if label == "spanish"]
    french_indices = [i for i, label in enumerate(labels) if label == "french"]
    
    # Shuffle indices
    random.shuffle(spanish_indices)
    random.shuffle(french_indices)
    
    # Calculate split sizes
    n_spanish_test = int(len(spanish_indices) * test_size)
    n_french_test = int(len(french_indices) * test_size)
    
    # Split indices
    spanish_test_indices = spanish_indices[:n_spanish_test]
    spanish_train_indices = spanish_indices[n_spanish_test:]
    
    french_test_indices = french_indices[:n_french_test]
    french_train_indices = french_indices[n_french_test:]
    
    # Combine indices
    test_indices = spanish_test_indices + french_test_indices
    train_indices = spanish_train_indices + french_train_indices
    
    # Create train and test sets
    train_words = [words[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_words = [words[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    return train_words, test_words, train_labels, test_labels

def stratified_k_fold(words, labels, k=5, random_state=42):
    """
    Create stratified k-fold indices.
    
    Parameters:
    -----------
    words : list of str
        The list of words
    labels : list of str
        The list of labels
    k : int
        Number of folds
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    list of tuples
        Each tuple contains (train_indices, test_indices)
    """
    # Set random seed
    random.seed(random_state)
    
    # Create indices for each class
    spanish_indices = [i for i, label in enumerate(labels) if label == "spanish"]
    french_indices = [i for i, label in enumerate(labels) if label == "french"]
    
    # Shuffle indices
    random.shuffle(spanish_indices)
    random.shuffle(french_indices)
    
    # Split indices into k folds
    spanish_folds = np.array_split(spanish_indices, k)
    french_folds = np.array_split(french_indices, k)
    
    # Create train-test splits for k-fold
    folds = []
    
    for i in range(k):
        # Create test indices for this fold
        test_indices = list(spanish_folds[i]) + list(french_folds[i])
        
        # Create train indices for this fold (all other folds)
        train_spanish_indices = [idx for j, fold in enumerate(spanish_folds) if j != i for idx in fold]
        train_french_indices = [idx for j, fold in enumerate(french_folds) if j != i for idx in fold]
        train_indices = train_spanish_indices + train_french_indices
        
        folds.append((train_indices, test_indices))
    
    return folds

def calculate_metrics(true_labels, predicted_labels):
    """
    Calculate evaluation metrics.
    
    Parameters:
    -----------
    true_labels : list
        The true labels
    predicted_labels : list
        The predicted labels
        
    Returns:
    --------
    dict
        A dictionary of metrics (accuracy, precision, recall, F1, confusion matrix)
    """
    # Get unique classes
    classes = np.unique(true_labels)
    
    # Initialize metrics
    metrics = {
        'accuracy': 0,
        'precision': {c: 0 for c in classes},
        'recall': {c: 0 for c in classes},
        'f1': {c: 0 for c in classes},
        'confusion_matrix': np.zeros((len(classes), len(classes)))
    }
    
    # Calculate accuracy
    correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    metrics['accuracy'] = correct / len(true_labels)
    
    # Build confusion matrix
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    for true, pred in zip(true_labels, predicted_labels):
        metrics['confusion_matrix'][class_to_idx[true]][class_to_idx[pred]] += 1
    
    # Calculate per-class metrics
    for c in classes:
        # True positives, false positives, false negatives
        tp = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == c and pred == c)
        fp = sum(1 for true, pred in zip(true_labels, predicted_labels) if true != c and pred == c)
        fn = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == c and pred != c)
        
        # Precision
        metrics['precision'][c] = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall
        metrics['recall'][c] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1
        precision = metrics['precision'][c]
        recall = metrics['recall'][c]
        metrics['f1'][c] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return metrics

def cross_validate(classify_func, words, labels, model_type='ensemble', use_tfidf=True, k=5, random_state=42):
    """
    Perform k-fold cross-validation.
    
    Parameters:
    -----------
    classify_func : function
        The classification function to use
    words : list of str
        The list of words
    labels : list of str
        The list of labels
    model_type : str
        The type of model to use
    use_tfidf : bool
        Whether to use TF-IDF weighting
    k : int
        Number of folds
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (avg_accuracy, std_accuracy, avg_time, fold_accuracies)
    """
    # Create k-fold indices
    folds = stratified_k_fold(words, labels, k, random_state)
    
    # Initialize results
    fold_accuracies = []
    fold_times = []
    fold_precisions = {'spanish': [], 'french': []}
    fold_recalls = {'spanish': [], 'french': []}
    fold_f1s = {'spanish': [], 'french': []}
    
    # Check if the classify function accepts model_type and use_tfidf parameters
    sig = inspect.signature(classify_func)
    param_count = len(sig.parameters)
    
    # Perform k-fold cross-validation
    for i, (train_indices, test_indices) in enumerate(folds):
        print(f"Fold {i+1}/{k}...")
        
        # Create train and test sets
        train_words = [words[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        test_words = [words[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]
        
        # Measure classification time
        start_time = time.time()
        
        # Classify test words
        if param_count >= 5:  # Function accepts all parameters
            predictions = classify_func(train_words, train_labels, test_words, model_type, use_tfidf)
        elif param_count == 3:  # Function only accepts basic parameters
            predictions = classify_func(train_words, train_labels, test_words)
        else:
            raise ValueError(f"Unsupported function signature for classify_func: {sig}")
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        fold_times.append(elapsed_time)
        
        # Calculate metrics
        metrics = calculate_metrics(test_labels, predictions)
        
        # Store metrics
        fold_accuracies.append(metrics['accuracy'])
        fold_precisions['spanish'].append(metrics['precision']['spanish'])
        fold_precisions['french'].append(metrics['precision']['french'])
        fold_recalls['spanish'].append(metrics['recall']['spanish'])
        fold_recalls['french'].append(metrics['recall']['french'])
        fold_f1s['spanish'].append(metrics['f1']['spanish'])
        fold_f1s['french'].append(metrics['f1']['french'])
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: Spanish={metrics['precision']['spanish']:.4f}, French={metrics['precision']['french']:.4f}")
        print(f"  Recall: Spanish={metrics['recall']['spanish']:.4f}, French={metrics['recall']['french']:.4f}")
        print(f"  F1: Spanish={metrics['f1']['spanish']:.4f}, French={metrics['f1']['french']:.4f}")
        print(f"  Time: {elapsed_time:.2f} seconds")
    
    # Calculate average results
    avg_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    avg_time = np.mean(fold_times)
    
    avg_precision = {
        'spanish': np.mean(fold_precisions['spanish']),
        'french': np.mean(fold_precisions['french'])
    }
    
    avg_recall = {
        'spanish': np.mean(fold_recalls['spanish']),
        'french': np.mean(fold_recalls['french'])
    }
    
    avg_f1 = {
        'spanish': np.mean(fold_f1s['spanish']),
        'french': np.mean(fold_f1s['french'])
    }
    
    print("\nCross-Validation Results:")
    print(f"Average accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average precision: Spanish={avg_precision['spanish']:.4f}, French={avg_precision['french']:.4f}")
    print(f"Average recall: Spanish={avg_recall['spanish']:.4f}, French={avg_recall['french']:.4f}")
    print(f"Average F1: Spanish={avg_f1['spanish']:.4f}, French={avg_f1['french']:.4f}")
    print(f"Average classification time: {avg_time:.2f} seconds")
    print(f"Fold accuracies: {', '.join(f'{acc:.4f}' for acc in fold_accuracies)}")
    
    return avg_accuracy, std_accuracy, avg_time, fold_accuracies

def evaluate_classifier(classify_func, file_path, words, labels, test_size=0.2, model_type='ensemble', use_tfidf=True, random_state=42):
    """
    Evaluate the classifier on the dataset with a train-test split.
    
    Parameters:
    -----------
    classify_func : function
        The classification function to use
    file_path : str
        Path to the data file (for display purposes)
    words : list of str
        The list of words
    labels : list of str
        The list of labels
    test_size : float
        Proportion of data to use for testing
    model_type : str
        Type of model to use
    use_tfidf : bool
        Whether to use TF-IDF weighting
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        A dictionary of results
    """
    # Split data into training and test sets
    train_words, test_words, train_labels, test_labels = train_test_split(
        words, labels, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set size: {len(train_words)}")
    print(f"Test set size: {len(test_words)}")
    print(f"Spanish words in training set: {train_labels.count('spanish')}")
    print(f"French words in training set: {train_labels.count('french')}")
    print(f"Spanish words in test set: {test_labels.count('spanish')}")
    print(f"French words in test set: {test_labels.count('french')}")
    
    # Measure classification time
    start_time = time.time()
    
    # Classify test words
    # Check if the classify function accepts model_type and use_tfidf parameters
    sig = inspect.signature(classify_func)
    param_count = len(sig.parameters)
    
    if param_count >= 5:  # Function accepts all parameters
        predictions = classify_func(train_words, train_labels, test_words, model_type, use_tfidf)
    elif param_count == 3:  # Function only accepts basic parameters
        predictions = classify_func(train_words, train_labels, test_words)
    else:
        raise ValueError(f"Unsupported function signature for classify_func: {sig}")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Calculate metrics
    metrics = calculate_metrics(test_labels, predictions)
    
    # Print results
    print("\nClassifier Results:")
    if model_type is not None:
        print(f"Model type: {model_type}")
    if use_tfidf is not None:
        print(f"Using TF-IDF: {use_tfidf}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: Spanish={metrics['precision']['spanish']:.4f}, French={metrics['precision']['french']:.4f}")
    print(f"Recall: Spanish={metrics['recall']['spanish']:.4f}, French={metrics['recall']['french']:.4f}")
    print(f"F1: Spanish={metrics['f1']['spanish']:.4f}, French={metrics['f1']['french']:.4f}")
    print(f"Classification Time: {elapsed_time:.2f} seconds")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("                Predicted Spanish  Predicted French")
    print(f"Actual Spanish    {metrics['confusion_matrix'][0][0]:16.0f}  {metrics['confusion_matrix'][0][1]:16.0f}")
    print(f"Actual French     {metrics['confusion_matrix'][1][0]:16.0f}  {metrics['confusion_matrix'][1][1]:16.0f}")
    
    # Analyze misclassifications
    misclassified_spanish = [word for word, true, pred in zip(test_words, test_labels, predictions) 
                         if true == 'spanish' and pred != 'spanish']
    misclassified_french = [word for word, true, pred in zip(test_words, test_labels, predictions) 
                        if true == 'french' and pred != 'french']
    
    print("\nMisclassified Words Analysis:")
    print(f"Total misclassified: {len(misclassified_spanish) + len(misclassified_french)} / {len(test_words)}")
    print(f"Misclassified Spanish words: {len(misclassified_spanish)} / {test_labels.count('spanish')}")
    print(f"Misclassified French words: {len(misclassified_french)} / {test_labels.count('french')}")
    
    if misclassified_spanish:
        print("\nSample misclassified Spanish words:")
        for word in misclassified_spanish[:10]:
            print(f"  {word}")
    
    if misclassified_french:
        print("\nSample misclassified French words:")
        for word in misclassified_french[:10]:
            print(f"  {word}")
    
    return {
        'metrics': metrics,
        'elapsed_time': elapsed_time,
        'predictions': predictions,
        'test_labels': test_labels,
        'test_words': test_words,
        'misclassified_spanish': misclassified_spanish,
        'misclassified_french': misclassified_french
    }

def analyze_feature_importance(feature_dict, model, top_n=20):
    """
    Analyze feature importance to identify the most discriminative features.
    
    Parameters:
    -----------
    feature_dict : dict
        The feature dictionary
    model : dict
        The trained model
    top_n : int
        Number of top features to return
        
    Returns:
    --------
    tuple
        (spanish_features, french_features)
    """
    if model['type'] != 'naive_bayes':
        print("Feature importance analysis is only supported for Naive Bayes models")
        return None, None
    
    # Reverse feature dictionary
    idx_to_feature = {idx: feature for feature, idx in feature_dict.items()}
    
    # Calculate feature importance as the ratio of likelihoods
    feature_importance = {}
    
    for idx, feature in idx_to_feature.items():
        spanish_likelihood = model['feature_likelihoods']['spanish'][idx]
        french_likelihood = model['feature_likelihoods']['french'][idx]
        
        if french_likelihood > 0 and spanish_likelihood > 0:
            ratio = spanish_likelihood / french_likelihood
            feature_importance[feature] = ratio
    
    # Get top features for each class
    spanish_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    french_features = sorted(feature_importance.items(), key=lambda x: x[1])[:top_n]
    
    # Print results
    print("\nFeature Importance Analysis:")
    
    print("\nTop Spanish-indicative features:")
    for feature, importance in spanish_features:
        print(f"  {feature}: {importance:.4f}")
    
    print("\nTop French-indicative features:")
    for feature, importance in french_features:
        print(f"  {feature}: {importance:.4f}")
    
    return spanish_features, french_features

def analyze_misclassifications(test_words, test_labels, predictions):
    """
    Analyze misclassified words to identify patterns.
    
    Parameters:
    -----------
    test_words : list of str
        The test words
    test_labels : list of str
        The true labels
    predictions : list of str
        The predicted labels
        
    Returns:
    --------
    tuple
        (misclassified_spanish, misclassified_french)
    """
    misclassified_spanish = []
    misclassified_french = []
    
    for word, true_label, pred_label in zip(test_words, test_labels, predictions):
        if true_label != pred_label:
            if true_label == "spanish":
                misclassified_spanish.append(word)
            else:
                misclassified_french.append(word)
    
    print("\nMisclassification Analysis:")
    print(f"Total misclassified words: {len(misclassified_spanish) + len(misclassified_french)}")
    print(f"Misclassified Spanish words: {len(misclassified_spanish)}")
    print(f"Misclassified French words: {len(misclassified_french)}")
    
    # Analyze common patterns in misclassified words
    if misclassified_spanish:
        print("\nSample misclassified Spanish words:")
        for word in misclassified_spanish[:10]:
            print(f"  {word}")
    
    if misclassified_french:
        print("\nSample misclassified French words:")
        for word in misclassified_french[:10]:
            print(f"  {word}")
    
    return misclassified_spanish, misclassified_french 