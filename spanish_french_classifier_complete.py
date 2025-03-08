import numpy as np
import pandas as pd
import re
import time
import random
from collections import Counter

# =====================================================================
# Core Classifier Functions
# =====================================================================

def preprocess_word(word):
    """
    Preprocess a word by converting to lowercase and removing any non-alphabetic characters.
    """
    # Convert to lowercase
    word = word.lower()
    # Remove any non-alphabetic characters
    word = re.sub(r'[^a-z]', '', word)
    return word

def extract_character_ngrams(word, n=1):
    """
    Extract character n-grams from a word.
    """
    ngrams = []
    for i in range(len(word) - n + 1):
        ngrams.append(word[i:i+n])
    return ngrams

def extract_position_specific_ngrams(word, n=1):
    """
    Extract position-specific n-grams from a word.
    """
    pos_ngrams = []
    
    # Beginning of word
    if len(word) >= n:
        pos_ngrams.append('BEGIN_' + word[:n])
    
    # End of word
    if len(word) >= n:
        pos_ngrams.append('END_' + word[-n:])
    
    return pos_ngrams

def extract_word_endings(word, n=2):
    """
    Extract the ending of a word.
    """
    if len(word) >= n:
        return word[-n:]
    return word

def extract_statistical_features(word):
    """
    Extract statistical features from a word.
    """
    features = []
    
    # Count vowels
    vowels = 'aeiou'
    vowel_count = sum(1 for char in word if char in vowels)
    
    # Vowel ratio
    if len(word) > 0:
        vowel_ratio = vowel_count / len(word)
        if vowel_ratio > 0.6:
            features.append('HIGH_VOWEL_RATIO')
        elif vowel_ratio < 0.3:
            features.append('LOW_VOWEL_RATIO')
    
    return features

def build_feature_dictionary(train_words, max_features=2000):
    """
    Build a dictionary of features from the training words.
    """
    # Extract all features from the training data
    all_features = []
    
    for word in train_words:
        word = preprocess_word(word)
        # Add unigrams
        all_features.extend(extract_character_ngrams(word, n=1))
        # Add bigrams
        all_features.extend(extract_character_ngrams(word, n=2))
        # Add trigrams
        all_features.extend(extract_character_ngrams(word, n=3))
        # Add position-specific n-grams
        all_features.extend(extract_position_specific_ngrams(word, n=1))
        all_features.extend(extract_position_specific_ngrams(word, n=2))
        # Add word endings
        all_features.append('ENDING_' + extract_word_endings(word, n=2))
        all_features.append('ENDING_' + extract_word_endings(word, n=3))
        # Add statistical features
        all_features.extend(extract_statistical_features(word))
        
        # Add specific vowel patterns based on data analysis
        for pattern in ['io', 'ua', 'eu', 'ou', 'ai', 'oi', 'ue', 'ie']:
            if pattern in word:
                all_features.append('PATTERN_' + pattern)
        
    # Count feature frequencies
    feature_counts = Counter(all_features)
    
    # Select the most common features
    most_common_features = feature_counts.most_common(max_features)
    
    # Create a dictionary mapping features to indices
    feature_dict = {}
    for i, (feature, _) in enumerate(most_common_features):
        feature_dict[feature] = i
    
    return feature_dict

def extract_features(word, feature_dict):
    """
    Extract features from a word using the feature dictionary.
    """
    word = preprocess_word(word)
    
    # Initialize feature vector
    feature_vector = np.zeros(len(feature_dict))
    
    # Extract unigrams
    for unigram in extract_character_ngrams(word, n=1):
        if unigram in feature_dict:
            feature_vector[feature_dict[unigram]] += 1
    
    # Extract bigrams
    for bigram in extract_character_ngrams(word, n=2):
        if bigram in feature_dict:
            feature_vector[feature_dict[bigram]] += 1
    
    # Extract trigrams
    for trigram in extract_character_ngrams(word, n=3):
        if trigram in feature_dict:
            feature_vector[feature_dict[trigram]] += 1
    
    # Extract position-specific n-grams
    for pos_ngram in extract_position_specific_ngrams(word, n=1):
        if pos_ngram in feature_dict:
            feature_vector[feature_dict[pos_ngram]] += 1
    
    for pos_ngram in extract_position_specific_ngrams(word, n=2):
        if pos_ngram in feature_dict:
            feature_vector[feature_dict[pos_ngram]] += 1
    
    # Extract word ending
    ending_feature_2 = 'ENDING_' + extract_word_endings(word, n=2)
    if ending_feature_2 in feature_dict:
        feature_vector[feature_dict[ending_feature_2]] += 1
    
    ending_feature_3 = 'ENDING_' + extract_word_endings(word, n=3)
    if ending_feature_3 in feature_dict:
        feature_vector[feature_dict[ending_feature_3]] += 1
    
    # Extract statistical features
    for stat_feature in extract_statistical_features(word):
        if stat_feature in feature_dict:
            feature_vector[feature_dict[stat_feature]] += 1
    
    # Add specific vowel patterns
    for pattern in ['io', 'ua', 'eu', 'ou', 'ai', 'oi', 'ue', 'ie']:
        if pattern in word and ('PATTERN_' + pattern) in feature_dict:
            feature_vector[feature_dict['PATTERN_' + pattern]] += 1
    
    return feature_vector

def train_naive_bayes(X_train, y_train, alpha=0.1):
    """
    Train a Multinomial Naive Bayes classifier.
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

def build_short_word_dictionary(train_words, train_labels):
    """
    Build a dictionary for short words (3 characters or less).
    """
    short_word_dict = {}
    
    for word, label in zip(train_words, train_labels):
        word = preprocess_word(word)
        if len(word) <= 3:
            short_word_dict[word] = label
    
    return short_word_dict

def classify(train_words, train_labels, test_words):
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
    
    Returns:
    --------
    list of str
        A list of predicted labels ("spanish" or "french") for test_words
    """
    try:
        # Preprocess the data
        preprocessed_train_words = [preprocess_word(word) for word in train_words]
        preprocessed_test_words = [preprocess_word(word) for word in test_words]
        
        # Build feature dictionary
        feature_dict = build_feature_dictionary(preprocessed_train_words, max_features=2000)
        
        # Build short word dictionary for special handling
        short_word_dict = build_short_word_dictionary(preprocessed_train_words, train_labels)
        
        # Extract features for training data
        X_train = np.array([extract_features(word, feature_dict) for word in preprocessed_train_words])
        
        # Extract features for test data
        X_test = np.array([extract_features(word, feature_dict) for word in preprocessed_test_words])
        
        # Train the model
        model = train_naive_bayes(X_train, train_labels, alpha=0.1)
        
        # Make predictions on test data
        predictions = []
        
        for i, word in enumerate(preprocessed_test_words):
            # Special handling for short words
            if len(word) <= 3 and word in short_word_dict:
                predictions.append(short_word_dict[word])
            else:
                # Use the model for prediction
                log_probs = {}
                for c in model['class_priors'].keys():
                    # Start with log prior
                    log_prob = np.log(model['class_priors'][c])
                    
                    # Add log likelihoods for each feature
                    for j, count in enumerate(X_test[i]):
                        if count > 0:
                            log_prob += count * np.log(model['feature_likelihoods'][c][j])
                    
                    log_probs[c] = log_prob
                
                # Predict the class with highest log probability
                predicted_class = max(log_probs, key=log_probs.get)
                predictions.append(predicted_class)
        
        return predictions
    except Exception as e:
        # In case of any error, return a default prediction
        print(f"Error in classification: {str(e)}")
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
    Split data into training and test sets.
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

def cross_validate(words, labels, k=5, random_state=42):
    """
    Perform k-fold cross-validation.
    """
    # Create indices for each class
    spanish_indices = [i for i, label in enumerate(labels) if label == "spanish"]
    french_indices = [i for i, label in enumerate(labels) if label == "french"]
    
    # Shuffle indices
    random.seed(random_state)
    random.shuffle(spanish_indices)
    random.shuffle(french_indices)
    
    # Split indices into k folds
    spanish_folds = np.array_split(spanish_indices, k)
    french_folds = np.array_split(french_indices, k)
    
    # Initialize results
    fold_accuracies = []
    fold_times = []
    
    # Perform k-fold cross-validation
    for i in range(k):
        print(f"Fold {i+1}/{k}...")
        
        # Create test indices for this fold
        test_indices = list(spanish_folds[i]) + list(french_folds[i])
        
        # Create train indices for this fold (all other folds)
        train_spanish_indices = [idx for j, fold in enumerate(spanish_folds) if j != i for idx in fold]
        train_french_indices = [idx for j, fold in enumerate(french_folds) if j != i for idx in fold]
        train_indices = train_spanish_indices + train_french_indices
        
        # Create train and test sets
        train_words = [words[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        test_words = [words[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]
        
        # Measure classification time
        start_time = time.time()
        
        # Classify test words
        predictions = classify(train_words, train_labels, test_words)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        fold_times.append(elapsed_time)
        
        # Calculate accuracy
        correct = sum(1 for pred, true in zip(predictions, test_labels) if pred == true)
        accuracy = correct / len(test_labels)
        fold_accuracies.append(accuracy)
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Time: {elapsed_time:.2f} seconds")
    
    # Calculate average results
    avg_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    avg_time = np.mean(fold_times)
    
    print("\nCross-Validation Results:")
    print(f"Average accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average classification time: {avg_time:.2f} seconds")
    print(f"Fold accuracies: {', '.join(f'{acc:.4f}' for acc in fold_accuracies)}")
    
    return avg_accuracy, std_accuracy, avg_time, fold_accuracies

def analyze_misclassifications(test_words, test_labels, predictions):
    """
    Analyze misclassified words to identify patterns.
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
        print("\nCommon patterns in misclassified Spanish words:")
        spanish_bigrams = []
        for word in misclassified_spanish:
            word = preprocess_word(word)
            spanish_bigrams.extend(extract_character_ngrams(word, n=2))
        
        spanish_bigram_counts = Counter(spanish_bigrams).most_common(5)
        print(f"Most common bigrams: {spanish_bigram_counts}")
    
    if misclassified_french:
        print("\nCommon patterns in misclassified French words:")
        french_bigrams = []
        for word in misclassified_french:
            word = preprocess_word(word)
            french_bigrams.extend(extract_character_ngrams(word, n=2))
        
        french_bigram_counts = Counter(french_bigrams).most_common(5)
        print(f"Most common bigrams: {french_bigram_counts}")
    
    return misclassified_spanish, misclassified_french

def analyze_feature_importance(train_words, train_labels):
    """
    Analyze feature importance to identify the most discriminative features.
    """
    # Preprocess the data
    preprocessed_train_words = [preprocess_word(word) for word in train_words]
    
    # Build feature dictionary
    feature_dict = build_feature_dictionary(preprocessed_train_words, max_features=2000)
    
    # Extract features for training data
    X_train = np.array([extract_features(word, feature_dict) for word in preprocessed_train_words])
    
    # Train the model
    model = train_naive_bayes(X_train, train_labels, alpha=0.1)
    
    # Calculate feature importance as the ratio of likelihoods
    feature_importance = {}
    for feature, idx in feature_dict.items():
        if "spanish" in model["feature_likelihoods"] and "french" in model["feature_likelihoods"]:
            spanish_likelihood = model["feature_likelihoods"]["spanish"][idx]
            french_likelihood = model["feature_likelihoods"]["french"][idx]
            
            if french_likelihood > 0:
                ratio = spanish_likelihood / french_likelihood
                feature_importance[feature] = ratio
    
    # Print most discriminative features
    print("\nFeature Importance Analysis:")
    
    print("\nTop Spanish-indicative features:")
    spanish_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for feature, importance in spanish_features:
        print(f"  {feature}: {importance:.4f}")
    
    print("\nTop French-indicative features:")
    french_features = sorted(feature_importance.items(), key=lambda x: x[1])[:10]
    for feature, importance in french_features:
        print(f"  {feature}: {importance:.4f}")
    
    return feature_importance

def evaluate_classifier(file_path="train (1).csv", test_size=0.2, random_state=42):
    """
    Evaluate the classifier on the dataset with a train-test split.
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
    
    # Print results
    print("\nClassifier Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Spanish Accuracy: {spanish_accuracy:.4f}")
    print(f"French Accuracy: {french_accuracy:.4f}")
    print(f"Classification Time: {elapsed_time:.2f} seconds")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("                Predicted Spanish  Predicted French")
    print(f"Actual Spanish    {spanish_correct:16d}  {len(spanish_indices) - spanish_correct:16d}")
    print(f"Actual French     {len(french_indices) - french_correct:16d}  {french_correct:16d}")
    
    # Analyze misclassifications
    misclassified_spanish, misclassified_french = analyze_misclassifications(
        test_words, test_labels, predictions
    )
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(train_words, train_labels)
    
    return {
        'accuracy': accuracy,
        'spanish_accuracy': spanish_accuracy,
        'french_accuracy': french_accuracy,
        'elapsed_time': elapsed_time,
        'predictions': predictions,
        'test_labels': test_labels,
        'test_words': test_words,
        'misclassified_spanish': misclassified_spanish,
        'misclassified_french': misclassified_french,
        'feature_importance': feature_importance
    }

# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    print("Spanish-French Word Classifier")
    print("=" * 50)
    
    print("\n1. Testing with 80/20 split:")
    results = evaluate_classifier(file_path="train (1).csv", test_size=0.2, random_state=42)
    
    print("\n2. Performing 5-fold cross-validation:")
    words, labels = load_data("train (1).csv")
    cross_validate(words, labels, k=5, random_state=42) 