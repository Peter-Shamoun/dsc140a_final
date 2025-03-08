import numpy as np
import pandas as pd
import re
import time
import random
from collections import Counter
from scipy.stats import chi2_contingency

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

def extract_feature_vector(word, feature_dict):
    """
    Extract features from a word.
    
    Parameters:
    -----------
    word : str
        The word to extract features from
    feature_dict : dict
        The feature dictionary
        
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
    
    return feature_vector

def extract_features_batch(words, feature_dict):
    """
    Extract features from a batch of words.
    
    Parameters:
    -----------
    words : list of str
        The words to extract features from
    feature_dict : dict
        The feature dictionary
        
    Returns:
    --------
    numpy.ndarray
        The feature matrix
    """
    # Initialize feature matrix
    X = np.zeros((len(words), len(feature_dict)))
    
    # Extract features for each word
    for i, word in enumerate(words):
        X[i] = extract_feature_vector(word, feature_dict)
    
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
        'class_priors': class_priors,
        'feature_likelihoods': feature_likelihoods
    }

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

def classify(train_words, train_labels, test_words):
    """
    Classify words as either Spanish or French using optimized Naive Bayes.
    
    Parameters:
    -----------
    train_words : list of str
        A list of words (in either Spanish or French) for training
    train_labels : list of str
        A list of labels ("spanish" or "french") corresponding to train_words
    test_words : list of str
        A list of words (in either Spanish or French) to classify
    
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
        
        # Build short word dictionary for special handling
        short_word_dict = build_short_word_dictionary(preprocessed_train_words, train_labels)
        
        # Extract features for test data
        X_test = extract_features_batch(preprocessed_test_words, feature_dict)
        
        # Train the Naive Bayes model
        model = train_naive_bayes(X_train, train_labels, alpha=0.1)
        
        # Make predictions
        predictions = []
        
        for i, word in enumerate(preprocessed_test_words):
            # Special handling for short words
            if len(word) <= 3 and word in short_word_dict:
                predictions.append(short_word_dict[word])
            else:
                # Use the model for prediction
                word_prediction = predict_naive_bayes(X_test[i:i+1], model)[0]
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

def evaluate_classifier(file_path="train (1).csv", test_size=0.2, random_state=42):
    """
    Evaluate the classifier on the dataset with a train-test split.
    
    Parameters:
    -----------
    file_path : str
        Path to the data file
    test_size : float
        Proportion of data to use for testing
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
    print(f"Spanish words in test set: {test_labels.count('spanish')}")
    print(f"French words in test set: {test_labels.count('french')}")
    
    # Measure classification time
    start_time = time.time()
    
    # Classify test words
    predictions = classify(train_words, train_labels, test_words)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Calculate accuracy
    correct = sum(1 for pred, true in zip(predictions, test_labels) if pred == true)
    accuracy = correct / len(test_labels)
    
    # Calculate language-specific accuracy
    spanish_indices = [i for i, label in enumerate(test_labels) if label == "spanish"]
    french_indices = [i for i, label in enumerate(test_labels) if label == "french"]
    
    spanish_correct = sum(1 for i in spanish_indices if predictions[i] == "spanish")
    french_correct = sum(1 for i in french_indices if predictions[i] == "french")
    
    spanish_accuracy = spanish_correct / len(spanish_indices) if spanish_indices else 0
    french_accuracy = french_correct / len(french_indices) if french_indices else 0
    
    # Calculate precision, recall, and F1 score
    spanish_precision = spanish_correct / sum(1 for pred in predictions if pred == "spanish") if sum(1 for pred in predictions if pred == "spanish") else 0
    french_precision = french_correct / sum(1 for pred in predictions if pred == "french") if sum(1 for pred in predictions if pred == "french") else 0
    
    spanish_recall = spanish_accuracy
    french_recall = french_accuracy
    
    spanish_f1 = 2 * (spanish_precision * spanish_recall) / (spanish_precision + spanish_recall) if (spanish_precision + spanish_recall) else 0
    french_f1 = 2 * (french_precision * french_recall) / (french_precision + french_recall) if (french_precision + french_recall) else 0
    
    # Print results
    print("\nClassifier Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Spanish Accuracy: {spanish_accuracy:.4f}")
    print(f"French Accuracy: {french_accuracy:.4f}")
    print(f"Spanish Precision: {spanish_precision:.4f}")
    print(f"French Precision: {french_precision:.4f}")
    print(f"Spanish Recall: {spanish_recall:.4f}")
    print(f"French Recall: {french_recall:.4f}")
    print(f"Spanish F1: {spanish_f1:.4f}")
    print(f"French F1: {french_f1:.4f}")
    print(f"Classification Time: {elapsed_time:.2f} seconds")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("                Predicted Spanish  Predicted French")
    print(f"Actual Spanish    {spanish_correct:16d}  {len(spanish_indices) - spanish_correct:16d}")
    print(f"Actual French     {len(french_indices) - french_correct:16d}  {french_correct:16d}")
    
    # Analyze misclassifications
    misclassified_spanish = [word for i, word in enumerate(test_words) if test_labels[i] == "spanish" and predictions[i] != "spanish"]
    misclassified_french = [word for i, word in enumerate(test_words) if test_labels[i] == "french" and predictions[i] != "french"]
    
    print("\nMisclassification Analysis:")
    print(f"Total misclassified words: {len(misclassified_spanish) + len(misclassified_french)}")
    print(f"Misclassified Spanish words: {len(misclassified_spanish)}")
    print(f"Misclassified French words: {len(misclassified_french)}")
    
    if misclassified_spanish:
        print("\nSample misclassified Spanish words:")
        for word in misclassified_spanish[:10]:
            print(f"  {word}")
    
    if misclassified_french:
        print("\nSample misclassified French words:")
        for word in misclassified_french[:10]:
            print(f"  {word}")
    
    return {
        'accuracy': accuracy,
        'spanish_accuracy': spanish_accuracy,
        'french_accuracy': french_accuracy,
        'spanish_precision': spanish_precision,
        'french_precision': french_precision,
        'spanish_recall': spanish_recall,
        'french_recall': french_recall,
        'spanish_f1': spanish_f1,
        'french_f1': french_f1,
        'elapsed_time': elapsed_time,
        'predictions': predictions,
        'test_labels': test_labels,
        'test_words': test_words,
        'misclassified_spanish': misclassified_spanish,
        'misclassified_french': misclassified_french
    }

def analyze_feature_importance(train_words, train_labels, top_n=20):
    """
    Analyze feature importance to identify the most discriminative features.
    
    Parameters:
    -----------
    train_words : list of str
        The training words
    train_labels : list of str
        The training labels
    top_n : int
        Number of top features to return
        
    Returns:
    --------
    tuple
        (spanish_features, french_features)
    """
    # Preprocess the data
    preprocessed_train_words = [preprocess_word(word, label) for word, label in zip(train_words, train_labels)]
    
    # Build feature dictionary
    feature_dict = build_feature_dictionary(preprocessed_train_words, train_labels, max_features=1500)
    
    # Extract features for training data
    X_train = extract_features_batch(preprocessed_train_words, feature_dict)
    
    # Train the model
    model = train_naive_bayes(X_train, train_labels, alpha=0.1)
    
    # Calculate feature importance as the ratio of likelihoods
    feature_importance = {}
    for feature, idx in feature_dict.items():
        if "spanish" in model["class_priors"] and "french" in model["class_priors"]:
            spanish_likelihood = model["feature_likelihoods"]["spanish"][idx]
            french_likelihood = model["feature_likelihoods"]["french"][idx]
            
            if french_likelihood > 0:
                ratio = spanish_likelihood / french_likelihood
                feature_importance[feature] = ratio
    
    # Get top features for each class
    spanish_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    french_features = sorted(feature_importance.items(), key=lambda x: x[1])[:top_n]
    
    # Print most discriminative features
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
    print("Spanish-French Word Classifier (Final Optimized Version)")
    print("=" * 70)
    
    print("\n1. Testing with 80/20 split:")
    results = evaluate_classifier(file_path="train (1).csv", test_size=0.2, random_state=42)
    
    print("\n2. Analyzing feature importance:")
    words, labels = load_data("train (1).csv")
    analyze_feature_importance(words, labels, top_n=20)
    
    # Compare with original implementation (if available)
    try:
        from spanish_french_classifier_complete import classify as original_classify
        from spanish_french_classifier_complete import load_data as original_load_data
        from spanish_french_classifier_complete import train_test_split as original_train_test_split
        
        print("\nComparing with Original Implementation:")
        
        # Load data
        words, labels = load_data("train (1).csv")
        
        # Split data using the same random state
        train_words, test_words, train_labels, test_labels = train_test_split(
            words, labels, test_size=0.2, random_state=42
        )
        
        # Measure classification time
        start_time = time.time()
        
        # Classify test words using original implementation
        original_predictions = original_classify(train_words, train_labels, test_words)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Calculate accuracy
        original_correct = sum(1 for pred, true in zip(original_predictions, test_labels) if pred == true)
        original_accuracy = original_correct / len(test_labels)
        
        print(f"\nAccuracy Comparison:")
        print(f"Original Implementation: {original_accuracy:.4f}")
        print(f"Optimized Implementation: {results['accuracy']:.4f}")
        
        improvement = results['accuracy'] - original_accuracy
        improvement_percent = improvement * 100
        
        print(f"Improvement: {improvement_percent:.2f}%")
        print(f"Speed Comparison: Original={elapsed_time:.2f}s, Optimized={results['elapsed_time']:.2f}s")
        
    except ImportError:
        print("\nOriginal implementation not available for comparison.")
    
    print("\nEnd of Analysis") 