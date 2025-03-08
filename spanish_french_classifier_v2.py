import numpy as np
import pandas as pd
import re
import time
import random
from collections import Counter
from scipy.stats import chi2_contingency
from sklearn.model_selection import KFold
import spanish_french_evaluation as eval

# =====================================================================
# Core Classifier Functions
# =====================================================================

def preprocess_word(word, language=None):
    """
    Enhanced preprocessing with language-specific normalization.
    
    Parameters:
    -----------
    word : str
        The word to preprocess
    language : str, optional
        If provided ('spanish' or 'french'), applies language-specific normalizations
        
    Returns:
    --------
    str
        The preprocessed word
    """
    # Convert to lowercase
    word = word.lower()
    
    # Remove any non-alphabetic characters but keep special characters for both languages
    word = re.sub(r'[^a-zñçàáâäèéêëìíîïòóôöùúûü]', '', word)
    
    # Handle language-specific normalizations if language is known (for training data)
    if language == 'spanish':
        # Normalize Spanish-specific patterns
        word = re.sub(r'qu[eéè]', 'ke', word)  # que -> ke
        word = re.sub(r'gu[eéè]', 'ge', word)  # gue -> ge
    elif language == 'french':
        # Normalize French-specific patterns
        word = re.sub(r'[œæ]', 'e', word)  # œ, æ -> e
        word = re.sub(r'ph', 'f', word)    # ph -> f
    
    return word

def extract_character_ngrams(word, n=1):
    """
    Extract character n-grams from a word.
    
    Parameters:
    -----------
    word : str
        The word to extract n-grams from
    n : int
        The size of the n-grams
        
    Returns:
    --------
    list of str
        A list of n-grams
    """
    ngrams = []
    for i in range(len(word) - n + 1):
        ngrams.append(word[i:i+n])
    return ngrams

def extract_position_specific_ngrams(word, n=1):
    """
    Extract position-specific n-grams from a word.
    
    Parameters:
    -----------
    word : str
        The word to extract position-specific n-grams from
    n : int
        The size of the n-grams
        
    Returns:
    --------
    list of str
        A list of position-specific n-grams
    """
    pos_ngrams = []
    
    # Beginning of word
    if len(word) >= n:
        pos_ngrams.append('BEGIN_' + word[:n])
    
    # End of word
    if len(word) >= n:
        pos_ngrams.append('END_' + word[-n:])
    
    return pos_ngrams

def extract_word_endings(word, max_length=3):
    """
    Extract multiple word endings of different lengths.
    
    Parameters:
    -----------
    word : str
        The word to extract endings from
    max_length : int
        The maximum ending length to extract
        
    Returns:
    --------
    list of str
        A list of word endings with their lengths
    """
    endings = []
    for n in range(1, max_length + 1):
        if len(word) >= n:
            endings.append(('ENDING_' + str(n), word[-n:]))
    return endings

def extract_language_specific_features(word):
    """
    Extract language-specific features that can help distinguish Spanish from French.
    
    Parameters:
    -----------
    word : str
        The word to extract language-specific features from
        
    Returns:
    --------
    list of str
        A list of language-specific features
    """
    features = []
    
    # Spanish-specific patterns
    spanish_patterns = {
        'ñ': 'SP_LETTER_Ñ',
        'll': 'SP_DIGRAPH_LL',
        'rr': 'SP_DIGRAPH_RR',
        'ción': 'SP_SUFFIX_CION',
        'dad': 'SP_SUFFIX_DAD',
        'ar$': 'SP_VERB_ENDING_AR',
        'er$': 'SP_VERB_ENDING_ER',
        'ir$': 'SP_VERB_ENDING_IR',
        'ía': 'SP_VOWEL_COMBO_IA',
        'ue': 'SP_VOWEL_COMBO_UE',
        'ie': 'SP_VOWEL_COMBO_IE'
    }
    
    # French-specific patterns
    french_patterns = {
        'ou': 'FR_VOWEL_COMBO_OU',
        'eu': 'FR_VOWEL_COMBO_EU',
        'au': 'FR_VOWEL_COMBO_AU',
        'eau$': 'FR_SUFFIX_EAU',
        'eux$': 'FR_SUFFIX_EUX',
        'tion$': 'FR_SUFFIX_TION',
        'ment$': 'FR_SUFFIX_MENT',
        'ç': 'FR_LETTER_C_CEDILLA',
        'è': 'FR_LETTER_E_GRAVE',
        'ê': 'FR_LETTER_E_CIRCUMFLEX',
        'ë': 'FR_LETTER_E_DIAERESIS',
        'é': 'FR_LETTER_E_ACUTE'
    }
    
    # Check for Spanish patterns
    for pattern, feature_name in spanish_patterns.items():
        if pattern.endswith('$'):
            if re.search(pattern, word):
                features.append(feature_name)
        elif pattern in word:
            features.append(feature_name)
    
    # Check for French patterns
    for pattern, feature_name in french_patterns.items():
        if pattern.endswith('$'):
            if re.search(pattern, word):
                features.append(feature_name)
        elif pattern in word:
            features.append(feature_name)
    
    # Add statistical features
    vowels = 'aeiouáéíóúàèìòùâêîôûäëïöü'
    vowel_count = sum(1 for char in word if char in vowels)
    consonants = sum(1 for char in word if char.isalpha() and char not in vowels)
    
    # Vowel ratio can be discriminative
    if len(word) > 0:
        vowel_ratio = vowel_count / len(word)
        if vowel_ratio > 0.6:
            features.append('HIGH_VOWEL_RATIO')
        elif vowel_ratio < 0.3:
            features.append('LOW_VOWEL_RATIO')
    
    # Consonant sequences can be discriminative
    consonant_seq = re.findall(r'[^aeiouáéíóúàèìòùâêîôûäëïöü]{3,}', word)
    if consonant_seq:
        features.append('LONG_CONSONANT_SEQ')
    
    return features

def extract_phonetic_features(word):
    """
    Extract phonetic features that capture pronunciation similarities.
    
    Parameters:
    -----------
    word : str
        The word to extract phonetic features from
        
    Returns:
    --------
    list of str
        A list of phonetic features
    """
    features = []
    
    # Create phonetic representation
    phonetic_word = word.lower()
    
    # Common transformations
    phonetic_word = re.sub(r'qu', 'k', phonetic_word)
    phonetic_word = re.sub(r'x', 'ks', phonetic_word)
    phonetic_word = re.sub(r'ç', 's', phonetic_word)
    phonetic_word = re.sub(r'[èéêë]', 'e', phonetic_word)
    phonetic_word = re.sub(r'[àâä]', 'a', phonetic_word)
    phonetic_word = re.sub(r'[ùûü]', 'u', phonetic_word)
    phonetic_word = re.sub(r'[îï]', 'i', phonetic_word)
    phonetic_word = re.sub(r'[ôö]', 'o', phonetic_word)
    
    # Extract phonetic n-grams
    for i in range(len(phonetic_word) - 1):
        features.append('PHON_' + phonetic_word[i:i+2])
    
    return features

def extract_all_features(word):
    """
    Extract all possible features from a word.
    
    Parameters:
    -----------
    word : str
        The word to extract features from
        
    Returns:
    --------
    list of str
        A list of all features
    """
    preprocessed_word = preprocess_word(word)
    
    if not preprocessed_word:
        return []
    
    features = []
    
    # Add character n-grams
    for n in range(1, 4):  # Unigrams, bigrams, trigrams
        features.extend(extract_character_ngrams(preprocessed_word, n))
    
    # Add position-specific n-grams
    for n in range(1, 3):  # Start/end unigrams and bigrams
        features.extend(extract_position_specific_ngrams(preprocessed_word, n))
    
    # Add word endings
    for ending_type, ending in extract_word_endings(preprocessed_word, max_length=3):
        features.append(f"{ending_type}_{ending}")
    
    # Add language-specific features
    features.extend(extract_language_specific_features(preprocessed_word))
    
    # Add phonetic features
    features.extend(extract_phonetic_features(preprocessed_word))
    
    return features

def build_feature_dictionary(train_words, train_labels, max_features=1500):
    """
    Build a feature dictionary with feature selection based on chi-squared statistics.
    
    Parameters:
    -----------
    train_words : list of str
        A list of words for training
    train_labels : list of str
        A list of labels corresponding to train_words
    max_features : int
        The maximum number of features to include
        
    Returns:
    --------
    dict
        A dictionary mapping features to indices
    """
    print("Building feature dictionary...")
    
    # Step 1: Extract all features from all words
    all_features = []
    for word in train_words:
        features = extract_all_features(word)
        all_features.extend(features)
    
    # Step 2: Count feature occurrences
    feature_counts = Counter(all_features)
    
    # Step 3: Create a matrix of feature occurrences per word
    feature_candidates = [feature for feature, count in feature_counts.items() if count >= 3]
    feature_to_index = {feature: i for i, feature in enumerate(feature_candidates)}
    
    # Initialize feature matrix (1 if feature is present in word, 0 otherwise)
    X = np.zeros((len(train_words), len(feature_candidates)), dtype=np.int8)
    
    # Fill the matrix
    for i, word in enumerate(train_words):
        word_features = set(extract_all_features(word))
        for feature in word_features:
            if feature in feature_to_index:
                X[i, feature_to_index[feature]] = 1
    
    # Step 4: Calculate chi-squared statistics for each feature
    classes = np.unique(train_labels)
    chi2_scores = np.zeros(len(feature_candidates))
    
    # Convert labels to binary for chi-squared calculation
    y_binary = np.array([1 if label == classes[0] else 0 for label in train_labels])
    
    print(f"Calculating chi-squared statistics for {len(feature_candidates)} features...")
    
    # Calculate chi-squared statistics for each feature
    for i in range(len(feature_candidates)):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{len(feature_candidates)} features...")
            
        # Create contingency table
        #  [feature present, class 0] [feature present, class 1]
        #  [feature absent, class 0]  [feature absent, class 1]
        contingency = np.array([
            [np.sum((X[:, i] == 1) & (y_binary == 0)), np.sum((X[:, i] == 1) & (y_binary == 1))],
            [np.sum((X[:, i] == 0) & (y_binary == 0)), np.sum((X[:, i] == 0) & (y_binary == 1))]
        ])
        
        # Skip features that are all zeros or all ones
        if np.any(contingency.sum(axis=1) == 0) or np.any(contingency.sum(axis=0) == 0):
            chi2_scores[i] = 0
            continue
            
        # Calculate chi-squared statistic
        chi2, _, _, _ = chi2_contingency(contingency, correction=False)
        chi2_scores[i] = chi2
    
    # Step 5: Select top features by chi-squared score
    selected_indices = np.argsort(chi2_scores)[-max_features:]
    selected_features = [feature_candidates[i] for i in selected_indices]
    
    # Build the final feature dictionary
    feature_dict = {}
    for i, feature in enumerate(selected_features):
        feature_dict[feature] = i
    
    print(f"Selected {len(feature_dict)} features out of {len(feature_candidates)} candidates.")
    
    return feature_dict

def compute_tf_idf_weights(X_train, feature_dict, train_words):
    """
    Compute TF-IDF weights for the features.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        The feature matrix
    feature_dict : dict
        The feature dictionary
    train_words : list of str
        The list of training words
        
    Returns:
    --------
    numpy.ndarray
        The IDF weights for each feature
    """
    # Number of documents
    n_samples = len(train_words)
    n_features = len(feature_dict)
    
    # Calculate document frequency for each feature
    document_freq = np.zeros(n_features)
    
    for i in range(n_samples):
        for j in range(n_features):
            if X_train[i, j] > 0:
                document_freq[j] += 1
    
    # Compute IDF values (add 1 for smoothing)
    idf = np.log(n_samples / (document_freq + 1)) + 1
    
    return idf

def extract_feature_vector(word, feature_dict, idf_weights=None):
    """
    Extract features from a word and optionally apply TF-IDF weighting.
    
    Parameters:
    -----------
    word : str
        The word to extract features from
    feature_dict : dict
        The feature dictionary
    idf_weights : numpy.ndarray, optional
        The IDF weights for TF-IDF transformation
        
    Returns:
    --------
    numpy.ndarray
        The feature vector
    """
    # Initialize feature vector
    feature_vector = np.zeros(len(feature_dict))
    
    # Extract features
    word_features = extract_all_features(word)
    
    # Count occurrences of each feature
    feature_counts = Counter(word_features)
    
    # Fill the feature vector
    for feature, count in feature_counts.items():
        if feature in feature_dict:
            feature_vector[feature_dict[feature]] = count
    
    # Apply TF-IDF weighting if provided
    if idf_weights is not None:
        feature_vector = feature_vector * idf_weights
    
    return feature_vector

def extract_features_batch(words, feature_dict, idf_weights=None):
    """
    Extract features from a batch of words.
    
    Parameters:
    -----------
    words : list of str
        The words to extract features from
    feature_dict : dict
        The feature dictionary
    idf_weights : numpy.ndarray, optional
        The IDF weights for TF-IDF transformation
        
    Returns:
    --------
    numpy.ndarray
        The feature matrix
    """
    # Initialize feature matrix
    X = np.zeros((len(words), len(feature_dict)))
    
    # Extract features for each word
    for i, word in enumerate(words):
        X[i] = extract_feature_vector(word, feature_dict, idf_weights)
    
    return X

def train_naive_bayes(X_train, y_train, alpha=0.1):
    """
    Train a Multinomial Naive Bayes classifier.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        The feature matrix
    y_train : list or numpy.ndarray
        The labels
    alpha : float
        The smoothing parameter
        
    Returns:
    --------
    dict
        The trained model
    """
    # Get unique classes
    classes = np.unique(y_train)
    
    # Initialize parameters
    class_priors = {}
    feature_likelihoods = {}
    
    # Calculate class priors and feature likelihoods
    for c in classes:
        # Get indices of samples in this class
        class_indices = np.array([i for i, label in enumerate(y_train) if label == c])
        
        # Calculate class prior
        class_priors[c] = len(class_indices) / len(y_train)
        
        # Get feature vectors for this class
        class_features = X_train[class_indices]
        
        # Calculate feature likelihoods with Laplace smoothing
        feature_counts = np.sum(class_features, axis=0) + alpha
        
        # Normalize to get probabilities
        total_counts = np.sum(feature_counts)
        feature_likelihoods[c] = feature_counts / total_counts
    
    return {
        'type': 'naive_bayes',
        'class_priors': class_priors,
        'feature_likelihoods': feature_likelihoods
    }

def train_linear_svm(X_train, y_train, learning_rate=0.01, epochs=100, C=1.0):
    """
    Train a Linear Support Vector Machine.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        The feature matrix
    y_train : list or numpy.ndarray
        The labels
    learning_rate : float
        The learning rate for gradient descent
    epochs : int
        The number of epochs for training
    C : float
        The regularization parameter
        
    Returns:
    --------
    dict
        The trained model
    """
    # Get unique classes
    classes = np.unique(y_train)
    
    if len(classes) != 2:
        raise ValueError("SVM implementation only supports binary classification")
    
    # Convert labels to binary: 1 for first class, -1 for second class
    y_binary = np.array([1 if label == classes[0] else -1 for label in y_train])
    
    # Initialize weights and bias
    n_features = X_train.shape[1]
    weights = np.zeros(n_features)
    bias = 0
    
    # Store best weights (with highest accuracy)
    best_weights = weights.copy()
    best_bias = bias
    best_accuracy = 0
    
    # Stochastic Gradient Descent
    n_samples = X_train.shape[0]
    indices = np.arange(n_samples)
    
    for epoch in range(epochs):
        # Shuffle data
        np.random.shuffle(indices)
        
        # Track progress
        if epoch % 10 == 0:
            # Calculate accuracy
            y_pred = np.sign(np.dot(X_train, weights) + bias)
            accuracy = np.mean(y_pred == y_binary)
            
            # Update best weights if accuracy improved
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = weights.copy()
                best_bias = bias
            
            print(f"Epoch {epoch}/{epochs}, accuracy: {accuracy:.4f}")
        
        # Train on each sample
        for i in indices:
            # Calculate current prediction
            margin = y_binary[i] * (np.dot(X_train[i], weights) + bias)
            
            # Update weights and bias
            if margin < 1:
                # Misclassified or within margin
                weights = weights + learning_rate * (C * y_binary[i] * X_train[i] - weights / n_samples)
                bias = bias + learning_rate * C * y_binary[i]
            else:
                # Correctly classified outside margin
                weights = weights - learning_rate * (weights / n_samples)
    
    # Return the best model
    return {
        'type': 'svm',
        'weights': best_weights,
        'bias': best_bias,
        'classes': classes
    }

def train_logistic_regression(X_train, y_train, learning_rate=0.01, epochs=100, reg_param=0.01):
    """
    Train a Logistic Regression classifier.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        The feature matrix
    y_train : list or numpy.ndarray
        The labels
    learning_rate : float
        The learning rate for gradient descent
    epochs : int
        The number of epochs for training
    reg_param : float
        The regularization parameter
        
    Returns:
    --------
    dict
        The trained model
    """
    # Get unique classes
    classes = np.unique(y_train)
    
    if len(classes) != 2:
        raise ValueError("Logistic Regression implementation only supports binary classification")
    
    # Convert labels to binary: 1 for first class, 0 for second class
    y_binary = np.array([1 if label == classes[0] else 0 for label in y_train])
    
    # Initialize weights and bias
    n_features = X_train.shape[1]
    weights = np.zeros(n_features)
    bias = 0
    
    # Define sigmoid function
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    # Store best weights (with highest accuracy)
    best_weights = weights.copy()
    best_bias = bias
    best_accuracy = 0
    
    # Stochastic Gradient Descent
    n_samples = X_train.shape[0]
    indices = np.arange(n_samples)
    
    for epoch in range(epochs):
        # Shuffle data
        np.random.shuffle(indices)
        
        # Track progress
        if epoch % 10 == 0:
            # Calculate predictions
            z = np.dot(X_train, weights) + bias
            y_pred_prob = sigmoid(z)
            y_pred = (y_pred_prob >= 0.5).astype(int)
            
            # Calculate accuracy
            accuracy = np.mean(y_pred == y_binary)
            
            # Update best weights if accuracy improved
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = weights.copy()
                best_bias = bias
            
            print(f"Epoch {epoch}/{epochs}, accuracy: {accuracy:.4f}")
        
        # Train on each sample
        for i in indices:
            # Calculate prediction
            z = np.dot(X_train[i], weights) + bias
            pred = sigmoid(z)
            
            # Calculate error
            error = y_binary[i] - pred
            
            # Update weights and bias with regularization
            weights = weights + learning_rate * (error * X_train[i] - reg_param * weights)
            bias = bias + learning_rate * error
    
    # Return the best model
    return {
        'type': 'logistic_regression',
        'weights': best_weights,
        'bias': best_bias,
        'classes': classes
    }

def train_ensemble(X_train, y_train):
    """
    Train an ensemble of classifiers.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        The feature matrix
    y_train : list or numpy.ndarray
        The labels
        
    Returns:
    --------
    dict
        The trained ensemble model
    """
    print("Training Naive Bayes classifier...")
    nb_model = train_naive_bayes(X_train, y_train, alpha=0.1)
    
    print("\nTraining SVM classifier...")
    svm_model = train_linear_svm(X_train, y_train, learning_rate=0.01, epochs=50, C=1.0)
    
    print("\nTraining Logistic Regression classifier...")
    lr_model = train_logistic_regression(X_train, y_train, learning_rate=0.01, epochs=50, reg_param=0.01)
    
    return {
        'type': 'ensemble',
        'models': [nb_model, svm_model, lr_model]
    }

def build_short_word_dictionary(train_words, train_labels):
    """
    Build a dictionary for short words (3 characters or less).
    
    Parameters:
    -----------
    train_words : list of str
        The training words
    train_labels : list of str
        The training labels
        
    Returns:
    --------
    dict
        A dictionary mapping short words to their labels
    """
    short_word_dict = {}
    
    for word, label in zip(train_words, train_labels):
        word = preprocess_word(word)
        if len(word) <= 3:
            short_word_dict[word] = label
    
    return short_word_dict

def predict_naive_bayes(X_test, model):
    """
    Make predictions using a Naive Bayes model.
    
    Parameters:
    -----------
    X_test : numpy.ndarray
        The test feature matrix
    model : dict
        The trained Naive Bayes model
        
    Returns:
    --------
    list
        The predicted labels
    """
    class_priors = model['class_priors']
    feature_likelihoods = model['feature_likelihoods']
    classes = list(class_priors.keys())
    
    # Initialize predictions
    predictions = []
    
    # Make predictions for each test instance
    for i in range(X_test.shape[0]):
        # Calculate log probabilities for each class
        log_probs = {}
        
        for c in classes:
            # Start with log prior
            log_prob = np.log(class_priors[c])
            
            # Add log likelihoods for each feature
            for j, count in enumerate(X_test[i]):
                if count > 0:
                    log_prob += count * np.log(feature_likelihoods[c][j])
            
            log_probs[c] = log_prob
        
        # Predict the class with highest log probability
        predicted_class = max(log_probs, key=log_probs.get)
        predictions.append(predicted_class)
    
    return predictions

def predict_linear_svm(X_test, model):
    """
    Make predictions using a Linear SVM model.
    
    Parameters:
    -----------
    X_test : numpy.ndarray
        The test feature matrix
    model : dict
        The trained SVM model
        
    Returns:
    --------
    list
        The predicted labels
    """
    weights = model['weights']
    bias = model['bias']
    classes = model['classes']
    
    # Calculate decision values
    decision_values = np.dot(X_test, weights) + bias
    
    # Convert to predictions
    binary_predictions = (decision_values >= 0).astype(int)
    
    # Map to class labels
    predictions = [classes[pred] for pred in binary_predictions]
    
    return predictions

def predict_logistic_regression(X_test, model):
    """
    Make predictions using a Logistic Regression model.
    
    Parameters:
    -----------
    X_test : numpy.ndarray
        The test feature matrix
    model : dict
        The trained Logistic Regression model
        
    Returns:
    --------
    list
        The predicted labels
    """
    weights = model['weights']
    bias = model['bias']
    classes = model['classes']
    
    # Define sigmoid function
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    # Calculate predictions
    z = np.dot(X_test, weights) + bias
    probabilities = sigmoid(z)
    binary_predictions = (probabilities >= 0.5).astype(int)
    
    # Map to class labels
    predictions = [classes[pred] for pred in binary_predictions]
    
    return predictions

def predict_ensemble(X_test, model):
    """
    Make predictions using an ensemble model.
    
    Parameters:
    -----------
    X_test : numpy.ndarray
        The test feature matrix
    model : dict
        The trained ensemble model
        
    Returns:
    --------
    list
        The predicted labels
    """
    models = model['models']
    
    # Make predictions using each individual model
    all_predictions = []
    
    for individual_model in models:
        if individual_model['type'] == 'naive_bayes':
            predictions = predict_naive_bayes(X_test, individual_model)
        elif individual_model['type'] == 'svm':
            predictions = predict_linear_svm(X_test, individual_model)
        elif individual_model['type'] == 'logistic_regression':
            predictions = predict_logistic_regression(X_test, individual_model)
        else:
            continue
        
        all_predictions.append(predictions)
    
    # Combine predictions using majority voting
    final_predictions = []
    
    for i in range(X_test.shape[0]):
        votes = {}
        for model_predictions in all_predictions:
            if i < len(model_predictions):  # Ensure index is in range
                label = model_predictions[i]
                votes[label] = votes.get(label, 0) + 1
        
        # Get the label with the most votes (or default to first class if tie)
        if votes:
            final_predictions.append(max(votes.items(), key=lambda x: x[1])[0])
        else:
            # Handle the case when no votes are available
            final_predictions.append('spanish')  # Default class
    
    return final_predictions

def classify(train_words, train_labels, test_words, model_type='ensemble', use_tfidf=True):
    """
    Classify words as either Spanish or French.
    
    Parameters:
    -----------
    train_words : list of str
        A list of words (in either Spanish or French) for training
    train_labels : list of str
        A list of labels ("spanish" or "french") corresponding to train_words
    test_words : list of str
        A list of words (in either Spanish or French) to classify
    model_type : str
        The type of model to use ('naive_bayes', 'svm', 'logistic_regression', or 'ensemble')
    use_tfidf : bool
        Whether to use TF-IDF weighting
        
    Returns:
    --------
    list of str
        A list of predicted labels ("spanish" or "french") for test_words
    """
    try:
        # Preprocess the training and test data
        preprocessed_train_words = [preprocess_word(word, label) for word, label in zip(train_words, train_labels)]
        preprocessed_test_words = [preprocess_word(word) for word in test_words]
        
        # Build feature dictionary
        feature_dict = build_feature_dictionary(preprocessed_train_words, train_labels, max_features=1500)
        
        # Extract features for training data
        X_train = extract_features_batch(preprocessed_train_words, feature_dict)
        
        # Compute TF-IDF weights if needed
        idf_weights = None
        if use_tfidf:
            idf_weights = compute_tf_idf_weights(X_train, feature_dict, preprocessed_train_words)
            X_train = X_train * idf_weights  # Apply TF-IDF weighting
        
        # Build short word dictionary for special handling
        short_word_dict = build_short_word_dictionary(preprocessed_train_words, train_labels)
        
        # Extract features for test data
        X_test = extract_features_batch(preprocessed_test_words, feature_dict, idf_weights)
        
        # Train the model
        if model_type == 'naive_bayes':
            model = train_naive_bayes(X_train, train_labels, alpha=0.1)
        elif model_type == 'svm':
            model = train_linear_svm(X_train, train_labels, learning_rate=0.01, epochs=50, C=1.0)
        elif model_type == 'logistic_regression':
            model = train_logistic_regression(X_train, train_labels, learning_rate=0.01, epochs=50, reg_param=0.01)
        elif model_type == 'ensemble':
            model = train_ensemble(X_train, train_labels)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Make predictions
        predictions = []
        
        for i, word in enumerate(preprocessed_test_words):
            # Special handling for short words
            if len(word) <= 3 and word in short_word_dict:
                predictions.append(short_word_dict[word])
            else:
                # Use the model for prediction
                if model_type == 'naive_bayes':
                    word_prediction = predict_naive_bayes(X_test[i:i+1], model)[0]
                elif model_type == 'svm':
                    word_prediction = predict_linear_svm(X_test[i:i+1], model)[0]
                elif model_type == 'logistic_regression':
                    word_prediction = predict_logistic_regression(X_test[i:i+1], model)[0]
                elif model_type == 'ensemble':
                    word_prediction = predict_ensemble(X_test[i:i+1], model)[0]
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                predictions.append(word_prediction)
        
        return predictions
    except Exception as e:
        # In case of any error, print the error and return a default prediction
        print(f"Error in classification: {str(e)}")
        import traceback
        traceback.print_exc()
        return ["french"] * len(test_words)

# =====================================================================
# Testing and Evaluation Functions
# =====================================================================

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    data = pd.read_csv(file_path)
    return data["word"].tolist(), data["label"].tolist()

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

def cross_validate(words, labels, model_type='ensemble', use_tfidf=True, k=5, random_state=42):
    """
    Perform k-fold cross-validation.
    
    Parameters:
    -----------
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
        predictions = classify(train_words, train_labels, test_words, model_type, use_tfidf)
        
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

def evaluate_classifier(file_path="train (1).csv", test_size=0.2, model_type='ensemble', use_tfidf=True, random_state=42):
    """
    Evaluate the classifier on the dataset with a train-test split.
    
    Parameters:
    -----------
    file_path : str
        Path to the data file
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
    # Load data
    words, labels = load_data(file_path)
    
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
    predictions = classify(train_words, train_labels, test_words, model_type, use_tfidf)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Calculate metrics
    metrics = calculate_metrics(test_labels, predictions)
    
    # Print results
    print("\nClassifier Results:")
    print(f"Model type: {model_type}")
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

# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    print("Spanish-French Word Classifier v2")
    print("=" * 50)
    
    # Load the data
    file_path = "train (1).csv"
    words, labels = load_data(file_path)
    
    print(f"Loaded {len(words)} words from {file_path}")
    print(f"Spanish words: {labels.count('spanish')}")
    print(f"French words: {labels.count('french')}")
    
    # Test with different models and configurations
    print("\n1. Testing with Ensemble Model (80/20 split):")
    results_ensemble = eval.evaluate_classifier(
        classify, 
        file_path, 
        words, 
        labels, 
        test_size=0.2, 
        model_type='ensemble', 
        use_tfidf=True, 
        random_state=42
    )
    
    print("\n2. Testing with Naive Bayes (80/20 split):")
    results_nb = eval.evaluate_classifier(
        classify, 
        file_path, 
        words, 
        labels, 
        test_size=0.2, 
        model_type='naive_bayes', 
        use_tfidf=True, 
        random_state=42
    )
    
    print("\n3. Testing with SVM (80/20 split):")
    results_svm = eval.evaluate_classifier(
        classify, 
        file_path, 
        words, 
        labels, 
        test_size=0.2, 
        model_type='svm', 
        use_tfidf=True, 
        random_state=42
    )
    
    print("\n4. Testing with Logistic Regression (80/20 split):")
    results_lr = eval.evaluate_classifier(
        classify, 
        file_path, 
        words, 
        labels, 
        test_size=0.2, 
        model_type='logistic_regression', 
        use_tfidf=True, 
        random_state=42
    )
    
    print("\n5. Performing 5-fold cross-validation with Ensemble Model:")
    cv_results_ensemble = eval.cross_validate(
        classify,
        words, 
        labels, 
        model_type='ensemble', 
        use_tfidf=True, 
        k=5, 
        random_state=42
    )
    
    # Print a summary of results
    print("\nSummary of Results (Accuracy):")
    print(f"Ensemble Model: {results_ensemble['metrics']['accuracy']:.4f}")
    print(f"Naive Bayes: {results_nb['metrics']['accuracy']:.4f}")
    print(f"SVM: {results_svm['metrics']['accuracy']:.4f}")
    print(f"Logistic Regression: {results_lr['metrics']['accuracy']:.4f}")
    print(f"Ensemble Model (Cross-Validation): {cv_results_ensemble[0]:.4f} ± {cv_results_ensemble[1]:.4f}")
    
    # Compare with original implementation (if available)
    try:
        from spanish_french_classifier_complete import classify as original_classify
        from spanish_french_classifier_complete import load_data as original_load_data
        
        print("\nComparing with Original Implementation:")
        original_words, original_labels = original_load_data(file_path)
        
        # Evaluate original implementation
        results_original = eval.evaluate_classifier(
            original_classify, 
            file_path, 
            original_words, 
            original_labels, 
            test_size=0.2, 
            model_type=None, 
            use_tfidf=None, 
            random_state=42
        )
        
        # Print comparison
        print("\nAccuracy Comparison:")
        print(f"Original Implementation: {results_original['metrics']['accuracy']:.4f}")
        print(f"Improved Implementation (Ensemble): {results_ensemble['metrics']['accuracy']:.4f}")
        
        improvement = results_ensemble['metrics']['accuracy'] - results_original['metrics']['accuracy']
        improvement_percent = improvement * 100
        
        print(f"Improvement: {improvement_percent:.2f}%")
        
    except ImportError:
        print("\nOriginal implementation not available for comparison.")
    
    print("\nEnd of Analysis") 