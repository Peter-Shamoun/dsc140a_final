import spanish_french_classifier_v2 as v2
import spanish_french_evaluation as eval

# Load the data
file_path = "train (1).csv"
words, labels = v2.load_data(file_path)

print("Spanish-French Word Classifier - Naive Bayes Tests")
print("=" * 50)
print(f"Loaded {len(words)} words from {file_path}")
print(f"Spanish words: {labels.count('spanish')}")
print(f"French words: {labels.count('french')}")

# Test Naive Bayes with different alpha values
print("\n1. Testing Naive Bayes with different alpha values:")
for alpha in [0.001, 0.01, 0.1, 0.5, 1.0]:
    # Create a modified classify function with the specific alpha value
    def classify_with_alpha(train_words, train_labels, test_words):
        return v2.classify(train_words, train_labels, test_words, 
                          model_type='naive_bayes', use_tfidf=True)
    
    # Temporarily modify the train_naive_bayes function to use the current alpha
    original_train_func = v2.train_naive_bayes
    
    # Create a wrapper function with the current alpha
    def train_with_alpha(X_train, y_train, alpha_not_used=None):
        return original_train_func(X_train, y_train, alpha=alpha)
    
    # Replace the function
    v2.train_naive_bayes = train_with_alpha
    
    # Evaluate
    print(f"\nAlpha = {alpha}:")
    results = eval.evaluate_classifier(
        classify_with_alpha, 
        file_path, 
        words, 
        labels, 
        test_size=0.2, 
        model_type=None, 
        use_tfidf=None, 
        random_state=42
    )
    
    # Restore the original function
    v2.train_naive_bayes = original_train_func

# Test with and without TF-IDF
print("\n2. Testing Naive Bayes with and without TF-IDF:")

# Without TF-IDF
def classify_without_tfidf(train_words, train_labels, test_words):
    return v2.classify(train_words, train_labels, test_words, 
                      model_type='naive_bayes', use_tfidf=False)

print("\nWithout TF-IDF:")
results_without_tfidf = eval.evaluate_classifier(
    classify_without_tfidf, 
    file_path, 
    words, 
    labels, 
    test_size=0.2, 
    model_type=None, 
    use_tfidf=None, 
    random_state=42
)

# With TF-IDF
def classify_with_tfidf(train_words, train_labels, test_words):
    return v2.classify(train_words, train_labels, test_words, 
                      model_type='naive_bayes', use_tfidf=True)

print("\nWith TF-IDF:")
results_with_tfidf = eval.evaluate_classifier(
    classify_with_tfidf, 
    file_path, 
    words, 
    labels, 
    test_size=0.2, 
    model_type=None, 
    use_tfidf=None, 
    random_state=42
)

# Test with different feature limits
print("\n3. Testing Naive Bayes with different feature counts:")
for max_features in [500, 1000, 1500, 2000, 3000]:
    # Create a modified build_feature_dictionary function
    original_build_func = v2.build_feature_dictionary
    
    # Create a wrapper function with the current max_features
    def build_with_max_features(train_words, train_labels, max_features_not_used=None):
        return original_build_func(train_words, train_labels, max_features=max_features)
    
    # Replace the function
    v2.build_feature_dictionary = build_with_max_features
    
    # Evaluate
    print(f"\nMax Features = {max_features}:")
    results = eval.evaluate_classifier(
        classify_with_tfidf, 
        file_path, 
        words, 
        labels, 
        test_size=0.2, 
        model_type=None, 
        use_tfidf=None, 
        random_state=42
    )
    
    # Restore the original function
    v2.build_feature_dictionary = original_build_func

print("\nEnd of Tests") 