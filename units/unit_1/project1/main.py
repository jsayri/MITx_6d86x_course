import project1 as p1
import utils
import numpy as np

#-------------------------------------------------------------------------------
# Data loading. There is no need to edit code in this section.
#-------------------------------------------------------------------------------

train_data = utils.load_data('reviews_train.tsv')
val_data = utils.load_data('reviews_val.tsv')
test_data = utils.load_data('reviews_test.tsv')

train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))

dictionary = p1.bag_of_words(train_texts)

train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary)
val_bow_features = p1.extract_bow_feature_vectors(val_texts, dictionary)
test_bow_features = p1.extract_bow_feature_vectors(test_texts, dictionary)

#-------------------------------------------------------------------------------
# Problem 5
#-------------------------------------------------------------------------------
# print("Section 5, Algorithm discussion\n")
# toy_features, toy_labels = toy_data = utils.load_toy_data('toy_data.tsv')
#
# T = 10
# L = 0.2
#
# thetas_perceptron = p1.perceptron(toy_features, toy_labels, T)
# thetas_avg_perceptron = p1.average_perceptron(toy_features, toy_labels, T)
# thetas_pegasos = p1.pegasos(toy_features, toy_labels, T, L)
#
# def plot_toy_results(algo_name, thetas):
#     print('theta for', algo_name, 'is', ', '.join(map(str,list(thetas[0]))))
#     print('theta_0 for', algo_name, 'is', str(thetas[1]))
#     utils.plot_toy_data(algo_name, toy_features, toy_labels, thetas)
#
# plot_toy_results('Perceptron', thetas_perceptron)
# plot_toy_results('Average Perceptron', thetas_avg_perceptron)
# plot_toy_results('Pegasos', thetas_pegasos)

#-------------------------------------------------------------------------------
# Problem 7
#-------------------------------------------------------------------------------
# print("\nSection 7, Classification and accuracy\n")
# T = 10
# L = 0.01
#
# pct_train_accuracy, pct_val_accuracy = \
#    p1.classifier_accuracy(p1.perceptron, train_bow_features,val_bow_features,train_labels,val_labels,T=T)
# print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
# print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))
#
# avg_pct_train_accuracy, avg_pct_val_accuracy = \
#    p1.classifier_accuracy(p1.average_perceptron, train_bow_features,val_bow_features,train_labels,val_labels,T=T)
# print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
# print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", avg_pct_val_accuracy))
#
# avg_peg_train_accuracy, avg_peg_val_accuracy = \
#    p1.classifier_accuracy(p1.pegasos, train_bow_features,val_bow_features,train_labels,val_labels,T=T,L=L)
# print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
# print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))

#-------------------------------------------------------------------------------
# Problem 8
#-------------------------------------------------------------------------------
# print("\nSection 8, Classification and accuracy\n")
# data = (train_bow_features, train_labels, val_bow_features, val_labels)
#
# # values of T and lambda to try
# Ts = [1, 5, 10, 15, 25, 50]
# Ls = [0.001, 0.01, 0.1, 1, 10]
#
# pct_tune_results = utils.tune_perceptron(Ts, *data)
# print('perceptron valid:', list(zip(Ts, pct_tune_results[1])))
# print('best = {:.4f}, T={:.4f}'.format(np.max(pct_tune_results[1]), Ts[np.argmax(pct_tune_results[1])]))
#
# avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)
# print('avg perceptron valid:', list(zip(Ts, avg_pct_tune_results[1])))
# print('best = {:.4f}, T={:.4f}'.format(np.max(avg_pct_tune_results[1]), Ts[np.argmax(avg_pct_tune_results[1])]))
#
# # fix values for L and T while tuning Pegasos T and L, respective
# fix_L = 0.01
# peg_tune_results_T = utils.tune_pegasos_T(fix_L, Ts, *data)
# print('Pegasos valid: tune T', list(zip(Ts, peg_tune_results_T[1])))
# print('best = {:.4f}, T={:.4f}'.format(np.max(peg_tune_results_T[1]), Ts[np.argmax(peg_tune_results_T[1])]))
#
# fix_T = Ts[np.argmax(peg_tune_results_T[1])]
# peg_tune_results_L = utils.tune_pegasos_L(fix_T, Ls, *data)
# print('Pegasos valid: tune L', list(zip(Ls, peg_tune_results_L[1])))
# print('best = {:.4f}, L={:.4f}'.format(np.max(peg_tune_results_L[1]), Ls[np.argmax(peg_tune_results_L[1])]))
#
# utils.plot_tune_results('Perceptron', 'T', Ts, *pct_tune_results)
# utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)
# utils.plot_tune_results('Pegasos', 'T', Ts, *peg_tune_results_T)
# utils.plot_tune_results('Pegasos', 'L', Ls, *peg_tune_results_L)

#-------------------------------------------------------------------------------
# Use the best method (perceptron, average perceptron or Pegasos) along with
# the optimal hyperparameters according to validation accuracies to test
# against the test dataset. The test data has been provided as
# test_bow_features and test_labels.
#-------------------------------------------------------------------------------

# Your code here
# Validation of the test set with the best classification algorithm
# best method: pegasos with T = 25 & L = 0.01
print("\nSection 8, Accuracy on test set\n")
T = 25
L = 0.01
avg_peg_train_accuracy, avg_peg_test_accuracy = \
   p1.classifier_accuracy(p1.pegasos, train_bow_features,test_bow_features,train_labels,test_labels,T=T,L=L)
print("{:33} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
print("{:33} {:.4f}".format("Test accuracy for Pegasos:", avg_peg_test_accuracy))


# --- Hinge loss, single evaluation ---
# print("\nLocal test from executions in project1\n")
# feature_vector = np.array([1, 2])
# label, theta, theta_0 = 1, np.array([-1, 1]), -0.2
# hloss = p1.hinge_loss_single(feature_vector, label, theta, theta_0)
# print (hloss)

# --- Test pegasus single step ---
# Test when theta and theta_0 == 0
# feature_vector = np.array([-0.26475382, -0.26902969, -0.04276542, -0.1188501, 0.3125307, 0.3105614,\
#                             0.29610234, 0.47825941, 0.20519536, 0.21865269])
# label = -1
# L = 0.7221618485058139
# eta = 0.005519419625906408
# theta = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
# theta_0 = 0
#
# p1.pegasos_single_step_update(feature_vector, label, L, eta, theta, theta_0)

# Test for prediction * label > 1
# feature_vector = np.array([-0.36194338, 0.44975533, 0.02675913, -0.22315544, -0.26165029, 0.27728528, \
#                             0.02749936, 0.39743416, -0.28536653, 0.04857445])
# label = -1
# L = 0.2821908705614756
# eta = 0.5535321581004718
# theta = np.array([-0.36148429, 0.33757187, 0.04933931, -0.28961002, -0.26722188, -0.14899803, \
#                    0.0021666, 0.29798112, 0.29783438, 0.04404272])
# theta_0 = -2.0174911915381766
#
# p1.pegasos_single_step_update(feature_vector, label, L, eta, theta, theta_0)


#-------------------------------------------------------------------------------
# Assign to best_theta, the weights (and not the bias!) learned by your most
# accurate algorithm with the optimal choice of hyperparameters.
#-------------------------------------------------------------------------------
print("\nSection 8, The most explanatory unigrams \n")

T = 25
L = 0.01
theta, theta_0 = p1.pegasos(train_bow_features, train_labels, T=T, L=L)
best_theta = theta # Your code here
wordlist   = [word for (idx, word) in sorted(zip(dictionary.values(), dictionary.keys()))]
sorted_word_features = utils.most_explanatory_word(best_theta, wordlist)
print("Most Explanatory Word Features")
print(sorted_word_features[:10])


#-------------------------------------------------------------------------------
# Remove stop words from the dictionary and evaluate the accuracy results for
# pegasos algorithm with the optimal choice of hyperparameters.
#-------------------------------------------------------------------------------
print("\nSection 9, Remove stopwords and evaluate accuracy \n")
# load the stopwords.txt file and exclude them from the dictionary of words
new_dictionary = p1.bag_of_words(train_texts, 'stopwords.txt')

train_bow_sw_features = p1.extract_bow_feature_vectors(train_texts, new_dictionary)
val_bow_sw_features = p1.extract_bow_feature_vectors(val_texts, new_dictionary)
test_bow_sw_features = p1.extract_bow_feature_vectors(test_texts, new_dictionary)

# execute algorithm with the a new train features
T = 25
L = 0.01
avg_peg_train_accuracy, avg_peg_test_accuracy = \
   p1.classifier_accuracy(p1.pegasos, train_bow_sw_features,test_bow_sw_features,train_labels,test_labels,T=T,L=L)
print("{:33} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
print("{:33} {:.4f}".format("Test accuracy for Pegasos:", avg_peg_test_accuracy))


#-------------------------------------------------------------------------------
# Update binary to counts features, include stopwords & evaluate the accuracy
# results for pegasos algorithm with the optimal choice of hyperparameters.
#-------------------------------------------------------------------------------
print("\nSection 9, Change binary for counts features and evaluate test accuracy \n")

# counts features for features vector
train_count_features = p1.extract_bow_feature_vectors(train_texts, new_dictionary, bin_f=False)
val_count_features = p1.extract_bow_feature_vectors(val_texts, new_dictionary, bin_f=False)
test_count_features = p1.extract_bow_feature_vectors(test_texts, new_dictionary, bin_f=False)

# execute algorithm with the a new train features
T = 25
L = 0.01
avg_peg_train_accuracy, avg_peg_test_accuracy = \
   p1.classifier_accuracy(p1.pegasos, train_count_features,test_count_features,train_labels,test_labels,T=T,L=L)
print("{:33} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
print("{:33} {:.4f}".format("Test accuracy for Pegasos:", avg_peg_test_accuracy))
