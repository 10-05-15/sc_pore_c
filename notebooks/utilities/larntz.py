def larntzPerlman(M1, M2, sample_size, alpha=0.05):
    """Larntz-Perlman procedure for testing correlation matrix equivalence (two matrices).
    
    notes:
         - Null (H0):  Starting assumption that the two correlation matrices
         represent the same underlying phenomenon. Depict similar patterns of
         correlations between variables.
         
         - Reject H0:  LP procedure assesses how likely it 
         is that the observed differences between the matrices could have occurred by random chance. 
         If the test statistic (max S value) is extreme compared to what's expected under the null
         hypothesis (unlikely due to random variation), we reject.

    args:
        M1 (numpy.ndarray): The first correlation matrix (p x p).
        M2 (numpy.ndarray): The second correlation matrix (p x p).
        sample_size (int): Sample size.
        alpha (float): Alpha parameter for significance determination (default: 0.05).

    returns:
        tuple:
            - hypothesis_accepted (bool): True if the null hypothesis is accepted.
                False: reject, data cannot be pooled
                True: accept, data can be pooled
            - p_values (numpy.ndarray): Matrix of p-values (uncorrected).
            - s_matrix (numpy.ndarray): S matrix.
            - overall_p_value (float): Overall p-value.
    """

    num_variables = M1.shape[0]  # make sure same dims

    # fisher z, extract upper trianlge
    fisher_z_values = np.arctanh(np.array([
        M1[np.triu_indices(num_variables, k=1)],
        M2[np.triu_indices(num_variables, k=1)]
    ]))

    # mean Z score, S, test statistic
    mean_z_score = np.mean(fisher_z_values, axis=0)  # mean across mtxs
    s_values = (sample_size - 3) * np.sum((fisher_z_values - mean_z_score) ** 2, axis=0) 
    test_statistic = np.max(s_values)

    # sidak correction and null hypoth test, dof=1
    sidak_alpha = (1 - alpha) ** (2 / (num_variables * (num_variables - 1)))
    hypothesis_accepted = test_statistic <= chi2.ppf(sidak_alpha, 1) 

    # outputting
    p_values = np.zeros((num_variables, num_variables))
    p_values[np.triu_indices(num_variables, k=1)] = 1 - chi2.cdf(s_values, 1) 
    p_values += p_values.T

    s_matrix = np.zeros((num_variables, num_variables))
    s_matrix[np.triu_indices(num_variables, k=1)] = s_values
    s_matrix += s_matrix.T

    overall_p_value = 1 - (chi2.cdf(test_statistic, 1)**((num_variables * (num_variables - 1)) / 2))

    return hypothesis_accepted, p_values, s_matrix, overall_p_value