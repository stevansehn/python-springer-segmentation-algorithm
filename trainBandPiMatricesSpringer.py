import numpy as np
import random
from sklearn.linear_model import LogisticRegression

def col_means(alist):

    means = []
    for i in range(len(alist.T)):
        means.append(np.mean(alist[:,i]))
    
    return np.array(means)

def trainBandPiMatricesSpringer(state_observation_values):

    ## Prelim

    number_of_states = 4

    ## Set pi_vector
    # The true value of the pi vector, which are the initial state
    # probabilities, are dependant on the heart rate of each PCG, and the
    # individual sound duration for each patient. Therefore, instead of setting
    # a patient-dependant pi_vector, simplify by setting all states as equally
    # probable:

    pi_vector = np.array((0.25,0.25,0.25,0.25))

    ## Train the logistic regression-based B_matrix:

    # Initialise the B_matrix as a 1x4 cell array. This is to hold the
    # coefficients of the trained logisitic regression model for each state.
    B_matrix = np.empty((1,number_of_states))
    B_matrix[:] = np.nan
    B_matrix = B_matrix.tolist()

    statei_values = np.empty((number_of_states,1))
    statei_values[:] = np.nan
    statei_values = statei_values.tolist()

    ## Fazer isso num loop
    # for PCGi in range(len(state_observation_values)):
    statei_values[0] = np.vstack((state_observation_values[0][0],state_observation_values[1][0],state_observation_values[2][0],state_observation_values[3][0],state_observation_values[4][0]))
    statei_values[1] = np.vstack((state_observation_values[0][1],state_observation_values[1][1],state_observation_values[2][1],state_observation_values[3][1],state_observation_values[4][1]))
    statei_values[2] = np.vstack((state_observation_values[0][2],state_observation_values[1][2],state_observation_values[2][2],state_observation_values[3][2],state_observation_values[4][2]))
    statei_values[3] = np.vstack((state_observation_values[0][3],state_observation_values[1][3],state_observation_values[2][3],state_observation_values[3][3],state_observation_values[4][3]))

    # In order to use Bayes' formula with the logistic regression derived
    # probabilities, we need to get the probability of seeing a specific
    # observation in the total training data set. This is the
    # 'total_observation_sequence', and the mean and covariance for each state
    # is found:

    total_observation_sequence = np.vstack((statei_values[0], statei_values[1], statei_values[2], statei_values[3]))
    total_obs_distribution = np.empty((2,1))
    total_obs_distribution[:] = np.nan
    total_obs_distribution = total_obs_distribution.tolist()
    total_obs_distribution[0] = col_means(total_observation_sequence)
    total_obs_distribution[1] = np.cov(total_observation_sequence, rowvar=False)

    for state in range(number_of_states):

        # Randomly select indices of samples from the other states not being 
        # learnt, in order to balance the two data sets. The code below ensures
        # that if class 1 is being learnt vs the rest, the number of the rest =
        # the number of class 1, evenly split across all other classes
        length_of_state_samples = len(statei_values[state])

        # Number of samples required from each of the other states:
        length_per_other_state = int(np.floor(length_of_state_samples/(number_of_states-1)))


        #If the length of the main class / (num states - 1) >
        #length(shortest other class), then only select
        #length(shortect other class) from the other states,
        #and (3* length) for main class
        min_length_other_class = np.inf

        for other_state in range(number_of_states):
            samples_in_other_state = len(statei_values[other_state])

            if(other_state == state):
                0
            else:
                min_length_other_class = min(min_length_other_class, samples_in_other_state)

        #This means there aren't enough samples in one of the
        #states to match the length of the main class being
        #trained:
        if( length_per_other_state > min_length_other_class):
            length_per_other_state = min_length_other_class

        training_data = np.empty((2,1))
        training_data[:] = np.nan
        training_data = training_data.tolist()

        for other_state in range(number_of_states):

            samples_in_other_state = len(statei_values[other_state])

            if(other_state == state):
                #Make sure you only choose (n-1)*3 *
                #length_per_other_state samples for the main
                #state, to ensure that the sets are balanced:
                indices = random.sample(list(np.random.permutation(samples_in_other_state)), k=length_per_other_state*(number_of_states-1))
                training_data[0] = statei_values[other_state][indices,:]
            else:

                indices = random.sample(list(np.random.permutation(samples_in_other_state)), k=length_per_other_state)
                state_data = statei_values[other_state][indices,:]
                training_data[1].append(state_data)

        training_data[1] = np.vstack(training_data[1][1:])

        # Label all the data:
        labels = np.ones((len(training_data[0]) + len(training_data[1]), 1))
        labels[:len(training_data[0])+1] = 2

        # Train the logisitic regression model for this state:
        all_data = np.vstack((training_data[0], training_data[1]))
        X, y = all_data, labels.ravel()
        clf = LogisticRegression(random_state=0, solver='lbfgs',
                                 multi_class='ovr').fit(X, y)
        B = np.vstack((clf.intercept_, clf.coef_.reshape(-1,1)))
        B_matrix[0][state] = B
    
    return B_matrix, pi_vector, total_obs_distribution