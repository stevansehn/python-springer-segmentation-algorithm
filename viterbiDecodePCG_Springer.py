import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from get_duration_distributions import *

def sigma(a):
    return 1/(1 + np.exp(-a))

def ind2sub(array_shape, ind):
    rows = (ind.astype('int') / array_shape[1])
    cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)

def viterbiDecodePCG_Springer(observation_sequence, pi_vector, b_matrix, total_obs_distribution, heartrate, systolic_time, Fs, figures=False):

    ## Preliminary
    springer_options = default_Springer_HSMM_options(SpringerOptions)

    T = len(observation_sequence)
    N = 4 # Number of states

    # Setting the maximum duration of a single state. This is set to an entire
    # heart cycle:
    max_duration_D = int(1*60/heartrate*Fs)

    # Initialising the variables that are needed to find the optimal state path along
    # the observation sequence.
    # delta_t(j), as defined on page 264 of Rabiner, is the best score (highest
    # probability) along a single path, at time t, which accounts for the first
    # t observations and ends in State s_j. In this case, the length of the
    # matrix is extended by max_duration_D samples, in order to allow the use
    # of the extended Viterbi algortithm:
    delta = np.ones((T + max_duration_D-1, N))*(-np.inf)

    # The argument that maximises the transition between states (this is
    # basically the previous state that had the highest transition probability
    # to the current state) is tracked using the psi variable.
    psi = np.zeros((T + max_duration_D-1, N)).astype(int)

    # An additional variable, that is not included on page 264 or Rabiner, is
    # the state duration that maximises the delta variable. This is essential
    # for the duration dependant HMM.
    psi_duration = np.zeros((T + max_duration_D-1, N)).astype(int)

    ## Setting up observation probs
    observation_probs = np.zeros((T,N))

    for n in range(N):

        # MLR gives P(state|obs)
        # Therefore, need Bayes to get P(o|state)
        # P(o|state) = P(state|obs) * P(obs) / P(states)
        # Where p(obs) is derived from a MVN distribution from all
        # observations, and p(states) is taken from the pi_vector:
        w = b_matrix[0][n].ravel()
        X = np.hstack((np.ones(len(observation_sequence)).reshape(-1,1),observation_sequence))
        pihat = sigma(w @ X.T)

        for t in range(T):

            x = observation_sequence[t,:]
            mu = total_obs_distribution[0]
            cov = total_obs_distribution[1] # = sigma (MATLAB)
            mvn = multivariate_normal(mu,cov) #create a multivariate Gaussian object with specified mean and covariance matrix
            Po_correction = mvn.pdf(x) #evaluate the probability density at x
            observation_probs[t,n] = (pihat[t]*Po_correction)/pi_vector[0]

    ## Setting up state duration probabilities, using Gaussian distributions:
    d_distributions, max_S1, min_S1, max_S2, min_S2, max_systole, min_systole, max_diastole, min_diastole = get_duration_distributions(heartrate, systolic_time)

    duration_probs = np.zeros((N,3*Fs))
    duration_sum = np.zeros((N))

    for state_j in range(N):
        for d in range(max_duration_D):

            if(state_j == 0):
                mvn = multivariate_normal(d_distributions[state_j][0], d_distributions[state_j][1])
                duration_probs[state_j,d] = mvn.pdf(d)

                if(d < min_S1 or d > max_S1):
                    duration_probs[state_j,d] = np.finfo(float).tiny

            elif(state_j == 2):
                mvn = multivariate_normal(d_distributions[state_j][0], d_distributions[state_j][1])
                duration_probs[state_j,d] = mvn.pdf(d)

                if(d < min_S2 or d > max_S2):
                    duration_probs[state_j,d] = np.finfo(float).tiny

            elif(state_j == 1):
                mvn = multivariate_normal(d_distributions[state_j][0], d_distributions[state_j][1])
                duration_probs[state_j,d] = mvn.pdf(d)

                if(d < min_systole or d > max_systole):
                    duration_probs[state_j,d] = np.finfo(float).tiny

            elif(state_j == 3):
                mvn = multivariate_normal(d_distributions[state_j][0], d_distributions[state_j][1])
                duration_probs[state_j,d] = mvn.pdf(d)

                if(d < min_diastole or d > max_diastole):
                    duration_probs[state_j,d] = np.finfo(float).tiny

        duration_sum[state_j] = sum(duration_probs[state_j,:])

    if(len(duration_probs)>3*Fs):
        duration_probs = duration_probs[:,:3*Fs]

    # if(figures):

    # #     plt.title('Name', 'Duration probabilities')

    #     plt.plot(duration_probs[0,:]/duration_sum[0],'b',linewidth=2.0, label='S1 Duration')

    #     plt.plot(duration_probs[1,:]/duration_sum[1],'r',linewidth=2.0, label='Systolic Duration')

    #     plt.plot(duration_probs[2,:]/duration_sum[2],'g',linewidth=2.0, label='S2 Duration')

    #     plt.plot(duration_probs[3,:]/duration_sum[3],'k',linewidth=2.0, label='Diastolic Duration')

    #     plt.legend()
    #     plt.show()

    ## Perform the actual Viterbi Recursion:
    qt = np.zeros(len(delta)).astype(int)

    ## Initialisation Step

    #Equation 32a and 69a, but leave out the probability of being in
    #state i for only 1 sample, as the state could have started before time t = 0.

    delta[0,:] = np.log(pi_vector) + np.log(observation_probs[0,:]) #first value is the probability of intially being in each state * probability of observation 1 coming from each state

    #Equation 32b
    psi[0,:] = -1

    # The state duration probabilities are now used.
    # Change the a_matrix to have zeros along the diagonal, therefore, only
    # relying on the duration probabilities and observation probabilities to
    # influence change in states:
    # This would only be valid in sequences where the transition between states
    # follows a distinct order.
    a_matrix = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])

    ## Run the core Viterbi algorith

    ## Recursion

        ## The Extended Viterbi algorithm:

        # Equations 33a and 33b and 69a, b, c etc:
        # again, ommitting the p(d), as state could have started before t = 1

        # This implementation extends the standard implementation of the
        # duration-dependant Viterbi algorithm by allowing the durations to
        # extend beyond the start and end of the time series, thereby allowing
        # states to "start" and "stop" outside of the recorded signal. This
        # addresses the issue of partial states at the beginning and end of the
        # signal being labelled as the incorrect state. For instance, a
        # short-duration diastole at the beginning of a signal looks a lot like
        # systole, and can lead to labelling errors.

        # t spans input 2 to T + max_duration_D:
    epsilon = 1e-256
    for t in range(1,T+max_duration_D-1):
        for j in range(N):
            for d in range(max_duration_D):

                # The start of the analysis window, which is the current time
                # step, minus d (the time horizon we are currently looking back),
                # plus 1. The analysis window can be seen to be starting one
                # step back each time the variable d is increased.
                # This is clamped to 1 if extending past the start of the
                # record, and T-1 is extending past the end of the record:
                start_t = t - d - 1
                if(start_t<0):
                    start_t = 0

                if(start_t > T-2):
                    start_t = T-2
                    

                # The end of the analysis window, which is the current time
                # step, unless the time has gone past T, the end of the record, in
                # which case it is truncated to T. This allows the analysis
                # window to extend past the end of the record, so that the
                # timing durations of the states do not have to "end" at the end
                # of the record.
                end_t = t+1
                if(t>T):
                    end_t = T

                # Find the max_delta and index of that from the previous step
                # and the transition to the current step:
                # This is the first half of the expression of equation 33a from
                # Rabiner:
                max_delta =    np.max(delta[start_t,:] + np.log(a_matrix[:,j]+epsilon))
                max_index = np.argmax(delta[start_t,:] + np.log(a_matrix[:,j]+epsilon))

                # Find the normalised probabilities of the observations over the
                # analysis window:
                probs = np.prod(observation_probs[start_t:end_t,j])

                # Find the normalised probabilities of the observations at only
                # the time point at the start of the time window:
                if(probs == 0):
                    probs = np.finfo(float).tiny
                emission_probs = np.log(probs)

                # Keep a running total of the emmission probabilities as the
                # start point of the time window is moved back one step at a
                # time. This is the probability of seeing all the observations
                # in the analysis window in state j:
                if(emission_probs == 0 or np.isnan(emission_probs)):
                    emission_probs = np.finfo(float).tiny

                # Find the total probability of transitioning from the last
                # state to this one, with the observations and being in the same
                # state for the analysis window. This is the duration-dependant
                # variation of equation 33a from Rabiner:
                #                 fprintf('log((duration_probs(j,d)./duration_sum(j))):%d\n',log((duration_probs(j,d)./duration_sum(j))));
                delta_temp = max_delta + emission_probs + np.log((duration_probs[j,d]/duration_sum[j]))

                # Unlike equation 33a from Rabiner, the maximum delta could come
                # from multiple d values, or from multiple size of the analysis
                # window. Therefore, only keep the maximum delta value over the
                # entire analysis window:
                # If this probability is greater than the last greatest,
                # update the delta matrix and the time duration variable:

                if(delta_temp > delta[t,j]):
                    delta[t,j] = delta_temp
                    psi[t,j] = max_index
                    psi_duration[t,j] = d

    if(springer_options.use_mex):
        ## Termination

        # For the extended case, need to find max prob after end of actual
        # sequence:

        # Find just the delta after the end of the actual signal
        temp_delta = delta[T:,:]
        #Find the maximum value in this section, and which state it is in:
        idx = np.argmax(temp_delta)
        pos,_ = ind2sub(temp_delta.shape, idx)
        pos = int(pos)

        # Change this position to the real position in delta matrix:
        pos = pos + T

        if (pos>delta.size):
            pos = delta.size

        # 1) Find the last most probable state
        # 2) From the psi matrix, find the most likely preceding state
        # 3) Find the duration of the last state from the psi_duration matrix
        # 4) From the onset to the offset of this state, set to the most likely state
        # 5) Repeat steps 2 - 5 until reached the beginning of the signal


        # The initial steps 1-4 are equation 34b in Rabiner. 1) finds P*, the most
        # likely last state in the sequence, 2) finds the state that precedes the
        # last most likely state, 3) finds the onset in time of the last state
        # (included due to the duration-dependancy) and 4) sets the most likely last
        # state to the q_t variable.

        # 1)
        state = np.argmax(delta[pos,:]) - 1 # -1 corrige para índices do Python

        # 2)
        offset = pos + 1
        preceding_state = psi[offset,state]

        # 3)
        # state_duration = psi_duration(offset, state);
        onset = int(offset - psi_duration[offset,state])

        # 4)
        qt[onset:offset] = state

        # The state is then updated to the preceding state, found above, which must
        # end when the last most likely state started in the observation sequence:
        state = preceding_state

        count = 0
        # While the onset of the state is larger than the maximum duration
        # specified:

        while(onset > 1):

            #2)
            offset = onset - 1
            #     offset_array(offset,1) = inf;
            preceding_state = psi[offset,state]
            #     offset_array(offset,2) = preceding_state;


            #3)
            #     state_duration = psi_duration(offset, state);
            onset = offset - psi_duration[offset,state] + 1

            #4)
            #     offset_array(onset:offset,3) = state;

            if(onset<1):
                onset = 0

            qt[onset:offset] = state
            state = preceding_state
            count = count + 1

            if(count > 1000):
                break

    qt = qt[:T]

    return qt, delta, psi