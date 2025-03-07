# Spanish-French Word Classifier: Comprehensive Experimentation Plan

## Introduction

This document outlines a systematic approach to evaluate and compare different machine learning models for classifying words as either Spanish or French. The goal is to achieve at least 75% accuracy on the test set, with potential to exceed 80% for extra credit. Rather than prematurely selecting a single approach, this plan enables objective comparison across multiple methods to determine the optimal solution.

The experimentation framework is designed to be:
- **Reproducible**: All experiments can be replicated with the same results
- **Systematic**: Testing proceeds in a structured manner to efficiently compare approaches
- **Comprehensive**: All promising approaches are evaluated fairly
- **Objective**: Decision criteria are clearly defined and measurable

This plan serves as a roadmap for implementing and evaluating various classification approaches covered in DSC 140A, focusing on models that can be implemented from scratch using only NumPy, Pandas, SciPy, and standard Python libraries.

## 1. Structured Testing Framework

### 1.1 Data Preprocessing Pipeline

To ensure fair comparison across all models, we will implement a consistent data preprocessing pipeline:

1. **Data Loading**:
   - Load the training data from the provided CSV file
   - Split into training words and corresponding labels
   - Verify data balance (approximately equal Spanish and French words)
   - Create a data summary (word count, average length, character distribution)

2. **Data Cleaning**:
   - Convert all words to lowercase
   - Remove any remaining special characters or digits
   - Verify that accents have been properly removed as mentioned in the assignment

3. **Train-Validation Split**:
   - Create a stratified split of the training data (80% train, 20% validation)
   - Ensure class balance is maintained in both splits
   - Save indices for reproducibility across experiments

4. **Data Analysis**:
   - Calculate basic statistics (word length distribution by language)
   - Identify most frequent characters in each language
   - Analyze character n-gram distributions
   - Generate visualizations to inform feature engineering

### 1.2 Feature Extraction Pipeline

We will implement a modular feature extraction pipeline that can be standardized or customized per model:

1. **Base Feature Extractors**:
   - Character frequency counter (unigrams)
   - Character n-gram extractor (bigrams, trigrams)
   - Word suffix/prefix extractor (last/first n characters)
   - Statistical feature calculator (word length, vowel ratio, etc.)

2. **Feature Transformation**:
   - Normalization options (by word length, min-max scaling)
   - Binary feature conversion (presence/absence)
   - Feature selection based on information gain

3. **Feature Combination**:
   - Concatenation of multiple feature types
   - Weighted feature combination
   - Feature interaction terms

4. **Feature Extraction API**:
   - Common interface for all feature extractors
   - Caching mechanism for efficient reuse
   - Serialization for consistent application to test data

### 1.3 Cross-Validation Strategy

To reliably estimate model performance and avoid overfitting:

1. **K-Fold Cross-Validation**:
   - Implement stratified 5-fold cross-validation
   - Ensure each fold maintains the class distribution
   - Use fixed random seed for reproducibility

2. **Validation Protocol**:
   - For each fold:
     - Train on k-1 folds
     - Validate on the remaining fold
     - Record performance metrics
   - Average metrics across all folds
   - Calculate standard deviation to assess stability

3. **Hyperparameter Tuning**:
   - Implement nested cross-validation for hyperparameter selection
   - Outer loop: 5-fold for performance estimation
   - Inner loop: 3-fold for hyperparameter tuning

### 1.4 Performance Metrics

We will track multiple metrics to comprehensively evaluate model performance:

1. **Primary Metrics**:
   - Accuracy (target: ≥75%, stretch goal: ≥80%)
   - Precision, Recall, and F1-score for each class
   - Macro-averaged F1-score across classes

2. **Secondary Metrics**:
   - Area Under ROC Curve (AUC)
   - Log loss (cross-entropy)
   - Training time and prediction time

3. **Confusion Matrix Analysis**:
   - Generate confusion matrices for each model
   - Identify patterns in misclassifications
   - Analyze word characteristics of misclassified examples

4. **Learning Curves**:
   - Plot training and validation metrics vs. training set size
   - Identify potential overfitting or underfitting
   - Determine if more data would be beneficial

## 2. Model-Specific Approaches

For each classification approach, we will implement and evaluate various configurations to determine the optimal model. All implementations will adhere to the assignment constraints, using only NumPy, Pandas, SciPy, and standard Python libraries.

### 2.1 Naive Bayes Classifier

Naive Bayes is a probabilistic classifier well-suited for text classification tasks due to its efficiency with high-dimensional feature spaces.

#### Implementation Details:
- **Variants to Test**:
  - Multinomial Naive Bayes (for count-based features)
  - Bernoulli Naive Bayes (for binary features)
  - Gaussian Naive Bayes (for continuous features)

#### Hyperparameters to Explore:
- **Smoothing Parameter (α)**: [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
  - Controls Laplace smoothing to handle unseen features
  - Higher values provide more regularization

#### Feature Variations:
- Character frequency counts (normalized and raw)
- Binary character presence/absence
- Character n-grams (n=1,2,3)
- Word-level features (length, vowel ratio)
- Suffix/prefix features (last/first 1-3 characters)

#### Implementation Verification Tests:
- Verify probability calculations sum to 1
- Check log-likelihood calculations for numerical stability
- Confirm correct handling of unseen features
- Test on simple synthetic examples with known outcomes

#### Expected Computational Complexity:
- **Training**: O(N×D) where N is number of samples, D is feature dimension
- **Prediction**: O(C×D) where C is number of classes (2 in this case)
- **Memory**: O(C×D) for storing class-conditional probabilities

### 2.2 K-Nearest Neighbors Classifier

KNN is a non-parametric method that classifies based on the majority class among the k nearest neighbors in the feature space.

#### Implementation Details:
- **Distance Metrics**:
  - Euclidean distance
  - Manhattan distance
  - Cosine similarity
  - Jaccard similarity (for binary features)

#### Hyperparameters to Explore:
- **K Values**: [1, 3, 5, 7, 9, 11, 15, 21]
  - Number of neighbors to consider
  - Odd values avoid ties in binary classification
- **Distance Weighting**: [uniform, inverse, inverse squared]
  - How to weight neighbor votes based on distance

#### Feature Variations:
- Character frequency vectors (normalized by word length)
- Binary n-gram presence/absence
- Dimensionality reduction via feature selection
- Custom distance metrics for character sequences

#### Implementation Verification Tests:
- Verify distance calculations with manual examples
- Check neighbor selection logic
- Test tie-breaking mechanisms
- Validate weighting functions

#### Expected Computational Complexity:
- **Training**: O(N) (just storing training data)
- **Prediction**: O(N×D) for brute force, can be optimized with data structures
- **Memory**: O(N×D) for storing all training examples

### 2.3 Support Vector Machine

SVM finds the hyperplane that best separates the classes in the feature space, maximizing the margin between them.

#### Implementation Details:
- **Optimization Methods**:
  - Gradient descent
  - Sequential Minimal Optimization (SMO)
  - Dual formulation with quadratic programming

#### Hyperparameters to Explore:
- **Regularization Parameter (C)**: [0.01, 0.1, 1.0, 10.0, 100.0]
  - Controls trade-off between margin width and training error
- **Kernel**: [linear, polynomial (degree 2, 3)]
  - Linear kernel for high-dimensional sparse data
  - Polynomial kernel for capturing feature interactions

#### Feature Variations:
- Normalized character frequency vectors
- Selected n-grams based on information gain
- Feature scaling (standardization, min-max scaling)

#### Implementation Verification Tests:
- Verify optimization convergence
- Check decision boundary on synthetic 2D data
- Validate margin calculations
- Test prediction logic

#### Expected Computational Complexity:
- **Training**: O(N²×D) to O(N³×D) depending on implementation
- **Prediction**: O(S×D) where S is number of support vectors
- **Memory**: O(N²) for kernel matrix in dual formulation

### 2.4 Perceptron Classifier

The perceptron is a simple linear classifier that can be effective for linearly separable problems.

#### Implementation Details:
- **Variants to Test**:
  - Standard Perceptron
  - Averaged Perceptron
  - Voted Perceptron

#### Hyperparameters to Explore:
- **Learning Rate**: [0.001, 0.01, 0.1, 0.5, 1.0]
  - Controls step size during weight updates
- **Maximum Iterations**: [10, 50, 100, 500, 1000]
  - Number of passes through the training data
- **Regularization**: [None, L1, L2] with strength [0.001, 0.01, 0.1]
  - Prevents overfitting by penalizing large weights

#### Feature Variations:
- Normalized character frequency vectors
- Binary feature representation
- Feature selection based on correlation with class

#### Implementation Verification Tests:
- Verify weight update logic
- Check convergence on linearly separable data
- Test prediction function
- Validate learning rate effects

#### Expected Computational Complexity:
- **Training**: O(N×D×I) where I is number of iterations
- **Prediction**: O(D) for dot product calculation
- **Memory**: O(D) for storing weight vector

## 3. Ablation Studies

Ablation studies will help us understand which components of our models contribute most to performance, guiding both feature engineering and model selection decisions.

### 3.1 Feature Importance Analysis

To identify which features contribute most to classification performance:

1. **Individual Feature Performance**:
   - Train models using each feature type individually
   - Rank features by their standalone predictive power
   - Identify the minimal set of features needed for acceptable performance

2. **Feature Removal Impact**:
   - Start with the full feature set
   - Remove one feature type at a time
   - Measure the performance drop to quantify importance
   - Create feature importance rankings for each model

3. **Information Gain Analysis**:
   - Calculate information gain for each feature with respect to the class labels
   - Rank features by information gain
   - Determine optimal feature selection thresholds

4. **Feature Type Comparison**:
   - Compare performance of different feature categories:
     - Character-level features vs. word-level statistics
     - Unigrams vs. bigrams vs. trigrams
     - Frequency-based vs. binary features
     - Prefix/suffix features vs. full word features

### 3.2 Model Complexity Analysis

To understand how model complexity affects overfitting and performance:

1. **Feature Dimensionality Study**:
   - Vary the number of features used (top-k by information gain)
   - Plot performance vs. feature count
   - Identify the optimal feature dimensionality for each model

2. **Model-Specific Complexity Analysis**:
   - **Naive Bayes**: Vary smoothing parameter
   - **KNN**: Vary k (number of neighbors)
   - **SVM**: Vary regularization strength and kernel complexity
   - **Perceptron**: Vary regularization and number of iterations

3. **Training Set Size Impact**:
   - Train models on increasing subsets of the training data
   - Plot learning curves (training and validation performance vs. training set size)
   - Identify models that perform well with limited data

4. **Bias-Variance Tradeoff Analysis**:
   - Decompose error into bias and variance components
   - Identify models with high bias (underfitting) vs. high variance (overfitting)
   - Adjust model complexity to optimize the tradeoff

### 3.3 Hyperparameter Sensitivity Analysis

To understand how sensitive each model is to hyperparameter choices:

1. **One-at-a-Time Sensitivity**:
   - Vary each hyperparameter while keeping others fixed
   - Plot performance vs. hyperparameter value
   - Identify parameters with the largest impact on performance

2. **Interaction Effects**:
   - Test combinations of hyperparameter values
   - Identify interactions between parameters
   - Create heatmaps to visualize parameter interactions

3. **Robustness Assessment**:
   - Measure performance stability across hyperparameter variations
   - Identify models that maintain consistent performance
   - Quantify sensitivity using coefficient of variation

4. **Optimal Ranges Determination**:
   - Narrow down optimal ranges for each hyperparameter
   - Focus subsequent tuning within these ranges
   - Document sensitivity findings for final model selection

## 4. Time-Efficient Testing Sequence

To maximize efficiency and allow early elimination of underperforming approaches, we will structure our experimentation in phases, with each phase informing the next.

### 4.1 Phase 1: Initial Feature and Model Screening

The first phase focuses on quickly evaluating a wide range of features and models with minimal hyperparameter tuning:

1. **Feature Screening**:
   - Implement all base feature extractors
   - Evaluate each feature type individually using a simple model (e.g., Naive Bayes)
   - Identify top-performing feature types for further exploration
   - Eliminate clearly underperforming feature types

2. **Model Screening**:
   - Implement baseline versions of all models
   - Test each model with default hyperparameters using the top feature types
   - Rank models by initial performance
   - Eliminate models that significantly underperform (>10% worse than the best)

3. **Quick Combinations**:
   - Test simple combinations of top-performing features
   - Evaluate feature-model compatibility
   - Identify promising directions for detailed exploration

**Expected Duration**: 1-2 days
**Decision Point**: Select top 2-3 models and top 3-4 feature types for detailed analysis

### 4.2 Phase 2: Focused Feature Engineering

The second phase explores feature engineering in depth for the most promising models:

1. **Feature Refinement**:
   - Optimize feature extraction parameters (e.g., n-gram length)
   - Test feature normalization and transformation techniques
   - Implement feature selection based on information gain
   - Explore feature combinations and interactions

2. **Feature-Model Pairing**:
   - Identify which feature types work best with each model
   - Customize feature sets for specific models
   - Optimize feature dimensionality for each model

3. **Cross-Validation Assessment**:
   - Implement full cross-validation for promising feature-model pairs
   - Evaluate stability across folds
   - Identify potential overfitting issues

**Expected Duration**: 2-3 days
**Decision Point**: Finalize feature engineering approach for each model

### 4.3 Phase 3: Hyperparameter Optimization

The third phase focuses on finding optimal hyperparameters for the most promising models:

1. **Coarse Grid Search**:
   - Test widely spaced hyperparameter values
   - Identify regions of promising performance
   - Eliminate clearly suboptimal parameter ranges

2. **Fine-Tuning**:
   - Conduct detailed search within promising parameter regions
   - Implement nested cross-validation for unbiased evaluation
   - Optimize for validation performance

3. **Model-Specific Optimization**:
   - Focus on the most sensitive parameters for each model
   - Address any model-specific issues (e.g., convergence for SVM)
   - Fine-tune regularization to prevent overfitting

**Expected Duration**: 2-3 days
**Decision Point**: Select optimal hyperparameters for each model

### 4.4 Phase 4: Final Model Selection and Validation

The final phase compares the best versions of each model to select the optimal approach:

1. **Final Performance Comparison**:
   - Evaluate fully optimized models using 5-fold cross-validation
   - Compare primary and secondary metrics
   - Assess computational efficiency and implementation complexity

2. **Ensemble Exploration**:
   - Test simple ensembles of top-performing models
   - Evaluate potential performance gains
   - Consider implementation complexity trade-offs

3. **Final Model Selection**:
   - Select the best overall model based on performance and complexity
   - Prepare final implementation
   - Document all decisions and findings

**Expected Duration**: 1-2 days
**Final Output**: Optimized classifier implementation ready for submission

### 4.5 Early Stopping Criteria

To avoid wasting time on unpromising approaches, we will implement the following early stopping criteria:

1. **Feature Type Elimination**:
   - Eliminate feature types that perform >5% worse than the best in initial screening
   - Stop exploration of feature variations that show no improvement after 3 iterations

2. **Model Elimination**:
   - Eliminate models that consistently perform >7% worse than the best model
   - Stop hyperparameter tuning if no improvement after 5 iterations

3. **Complexity-Performance Tradeoff**:
   - Eliminate approaches that are significantly more complex but offer <2% performance gain
   - Prioritize simpler models when performance differences are small (<1%)

## 5. Visualization and Logging Strategy

To effectively track and analyze experimental results, we will implement a comprehensive visualization and logging system.

### 5.1 Experiment Tracking Framework

We will create a structured system to log all experimental details and results:

1. **Experiment Registry**:
   - Unique identifier for each experiment
   - Timestamp and duration
   - Model type and configuration
   - Feature set description
   - Hyperparameter values
   - Performance metrics

2. **Automated Logging**:
   - Implement a decorator for experiment functions
   - Automatically capture inputs, outputs, and performance
   - Record training and validation metrics for each fold
   - Track resource usage (memory, computation time)

3. **Experiment Database**:
   - Store all experiment results in a structured format (CSV or JSON)
   - Enable filtering and sorting by any parameter
   - Support querying for specific configurations or performance thresholds

4. **Version Control Integration**:
   - Link experiments to specific code versions
   - Track changes in implementation between experiments
   - Enable reproducibility of any experiment

### 5.2 Performance Visualizations

We will create standardized visualizations to analyze model performance:

1. **Metric Comparison Charts**:
   - Bar charts comparing models across key metrics
   - Radar charts for multi-metric comparison
   - Box plots showing performance distribution across folds

2. **Learning Curves**:
   - Training and validation metrics vs. epochs/iterations
   - Performance vs. training set size
   - Performance vs. feature count

3. **Confusion Matrix Visualizations**:
   - Heatmap representation of confusion matrices
   - Normalized and raw count versions
   - Highlighting of commonly confused word patterns

4. **Error Analysis Visualizations**:
   - Word clouds of misclassified words
   - Character distribution in errors vs. correct classifications
   - Length distribution of misclassified words

### 5.3 Feature Analysis Visualizations

To understand feature importance and relationships:

1. **Feature Importance Plots**:
   - Bar charts of feature importance scores
   - Cumulative importance curves
   - Feature importance by category (n-grams, statistics, etc.)

2. **Feature Distribution Visualizations**:
   - Histograms of feature values by class
   - Violin plots for comparing distributions
   - Scatter plots for feature pairs with class coloring

3. **Dimensionality Reduction Plots**:
   - PCA or t-SNE visualization of feature space
   - Class separation visualization
   - Decision boundary visualization (for applicable models)

4. **Character/N-gram Analysis**:
   - Heatmaps of character frequencies by language
   - Differential analysis of n-gram importance
   - Position-specific character importance

### 5.4 Hyperparameter Analysis Visualizations

To understand hyperparameter effects and interactions:

1. **Parameter Response Curves**:
   - Line plots of performance vs. parameter value
   - Confidence intervals across cross-validation folds
   - Multiple metrics on the same plot

2. **Parameter Interaction Heatmaps**:
   - 2D heatmaps for pairs of hyperparameters
   - 3D surface plots for critical parameter pairs
   - Contour plots of performance landscapes

3. **Regularization Effect Visualizations**:
   - Training vs. validation performance across regularization strengths
   - Model coefficient magnitudes vs. regularization
   - Overfitting visualization at different regularization levels

### 5.5 Progress Tracking Dashboard

To maintain an overview of the experimentation process:

1. **Summary Statistics Dashboard**:
   - Best performance achieved so far
   - Performance improvement over time
   - Resource usage and efficiency metrics
   - Progress through testing phases

2. **Comparative Analysis Views**:
   - Side-by-side comparison of top models
   - Performance delta from baseline
   - Trade-off visualization (performance vs. complexity)

3. **Timeline Visualization**:
   - Experiment history with performance milestones
   - Branching visualization of exploration paths
   - Highlighting of breakthrough moments

## 6. Statistical Significance Testing

To ensure that observed performance differences between models are reliable and not due to random chance, we will implement rigorous statistical testing methods.

### 6.1 Cross-Validation Significance Testing

For comparing model performance across cross-validation folds:

1. **Paired t-tests**:
   - Compare models using paired t-tests on fold-specific performance
   - Calculate p-values to determine if differences are statistically significant
   - Use Bonferroni correction for multiple comparisons
   - Require p < 0.05 for claiming significant differences

2. **Effect Size Calculation**:
   - Calculate Cohen's d for quantifying the magnitude of differences
   - Interpret effect sizes: small (0.2), medium (0.5), large (0.8)
   - Prioritize models with both statistically significant and meaningful effect sizes

3. **Confidence Intervals**:
   - Compute 95% confidence intervals for all performance metrics
   - Consider overlapping confidence intervals when comparing models
   - Report confidence intervals alongside point estimates

### 6.2 Bootstrap Resampling

To obtain more robust estimates of model performance:

1. **Bootstrap Performance Estimation**:
   - Generate 1000 bootstrap samples from the validation set
   - Evaluate models on each bootstrap sample
   - Calculate mean, median, and percentile intervals (2.5%, 97.5%)

2. **Bootstrap Hypothesis Testing**:
   - Use bootstrap to estimate the distribution of performance differences
   - Calculate the proportion of bootstrap samples where one model outperforms another
   - Require >95% of samples to show improvement for claiming significance

3. **Bootstrap Stability Analysis**:
   - Assess the stability of model rankings across bootstrap samples
   - Identify models with consistently high rankings
   - Quantify uncertainty in model selection

### 6.3 McNemar's Test for Classification Agreement

To compare the patterns of errors between models:

1. **Contingency Table Analysis**:
   - Create 2×2 contingency tables for pairs of models
   - Count cases where both models are correct, both wrong, or one correct and one wrong
   - Apply McNemar's test to assess if error patterns differ significantly

2. **Error Correlation Analysis**:
   - Calculate correlation between model errors
   - Identify models with uncorrelated errors (potential ensemble candidates)
   - Visualize error agreement patterns

3. **Word-Level Agreement Analysis**:
   - Analyze which types of words show disagreement between models
   - Identify complementary strengths of different models
   - Use findings to guide ensemble construction

### 6.4 Permutation Tests for Feature Importance

To validate the significance of feature importance measures:

1. **Null Hypothesis Testing**:
   - Randomly permute class labels to break the relationship with features
   - Calculate feature importance on permuted data
   - Compare actual importance to the null distribution
   - Identify features with significance level p < 0.05

2. **Feature Group Significance**:
   - Test the collective significance of feature groups (n-grams, statistics, etc.)
   - Use permutation tests to assess group importance
   - Identify which feature categories contribute significantly

3. **Stability of Feature Rankings**:
   - Assess consistency of feature rankings across resamples
   - Calculate rank correlation coefficients
   - Identify features with stable importance across variations

## 7. Final Model Selection Criteria

To select the optimal model for submission, we will use a comprehensive set of criteria that balance performance, robustness, and implementation considerations.

### 7.1 Primary Selection Criteria

The following criteria will be weighted most heavily in the final decision:

1. **Cross-Validation Performance**:
   - Mean accuracy across 5-fold cross-validation
   - Standard deviation of accuracy (lower is better for stability)
   - Performance on challenging word subsets (short words, ambiguous words)

2. **Statistical Significance**:
   - Models must demonstrate statistically significant improvement over baselines
   - Performance differences between top models must be significant (p < 0.05)
   - Bootstrap confidence intervals should support performance claims

3. **Generalization Assessment**:
   - Gap between training and validation performance (smaller is better)
   - Performance stability across different data subsets
   - Robustness to potential distribution shifts

### 7.2 Secondary Selection Criteria

These criteria will be considered when primary criteria are similar:

1. **Implementation Complexity**:
   - Code simplicity and readability
   - Number of hyperparameters requiring tuning
   - Computational efficiency (training and prediction time)
   - Memory requirements

2. **Feature Engineering Requirements**:
   - Complexity of feature extraction pipeline
   - Number of feature types required
   - Preprocessing dependencies

3. **Robustness Properties**:
   - Sensitivity to hyperparameter choices
   - Performance degradation with reduced feature sets
   - Stability across random initializations (if applicable)

### 7.3 Decision Framework

To make the final selection in a systematic way:

1. **Scoring System**:
   - Assign weights to each criterion based on importance
   - Score each model on a scale of 1-10 for each criterion
   - Calculate weighted sum for overall ranking

2. **Elimination Process**:
   - Start with all models that achieve >75% cross-validation accuracy
   - Eliminate models with statistically inferior performance
   - Among statistically tied models, prefer simpler implementations

3. **Tie-Breaking Strategy**:
   - In case of ties, prioritize models with:
     1. Lower variance across cross-validation folds
     2. Better performance on challenging word subsets
     3. Faster prediction time
     4. Simpler implementation

### 7.4 Final Verification

Before finalizing the selection:

1. **Code Review**:
   - Ensure implementation correctness
   - Verify that all constraints are met (no external libraries)
   - Check for potential bugs or edge cases

2. **Documentation Completeness**:
   - Document all design decisions
   - Explain feature engineering approach
   - Describe hyperparameter choices

3. **Final Cross-Check**:
   - Verify that the selected model consistently achieves target accuracy
   - Ensure that the model generalizes well to unseen examples
   - Confirm that the implementation is efficient and robust

## 8. Conclusion

This comprehensive experimentation plan provides a systematic approach to developing an optimal Spanish-French word classifier. By following this structured methodology, we can:

1. **Explore the solution space thoroughly** through systematic evaluation of different models, features, and hyperparameters
2. **Make evidence-based decisions** using rigorous statistical testing and performance metrics
3. **Efficiently allocate resources** by prioritizing promising approaches and eliminating underperforming ones early
4. **Document the development process** with detailed logging and visualization
5. **Select the optimal model** based on clear, objective criteria

The plan is designed to be both comprehensive and practical, allowing for thorough exploration while maintaining focus on the ultimate goal of achieving at least 75% accuracy on the test set, with potential to exceed 80% for extra credit.

By implementing this plan, we can develop a robust classifier that effectively distinguishes between Spanish and French words based solely on their character patterns, demonstrating a practical application of the machine learning concepts covered in DSC 140A. 