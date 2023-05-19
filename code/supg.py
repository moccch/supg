import numpy as np
import re
import matplotlib.pyplot as plt

n_records = 10**6

def parse_query(query):
    pattern = r"SELECT \* FROM (\w+) WHERE (.+) ORACLE LIMIT (\d+) USING proxy_estimates \[(RECALL|PRECISION)\] TARGET (\d+(\.\d+)?) WITH PROBABILITY (\d+(\.\d+)?)"
    match = re.match(pattern, query)
    if match:
        table_name, filter_predicate, o, proxy_extimates, t, _, p, _ = match.groups()
        return {
            "table_name": table_name,
            "filter_predicate": int(filter_predicate),
            "s": int(o),
            "proxy_extimates": proxy_extimates,
            "target": float(t),
            "delta": float(1- float(p))
        }
    else:
        return "Invalid query"

def UB(mu, sigma, s, delta):
    return mu + sigma * np.sqrt(2 / s * np.log(1/delta))

def LB(mu, sigma, s, delta):
    return mu - sigma * np.sqrt(2 / s * np.log(1/delta))

def RecallSw(tau, proxy_scores, oracle_labels, weights):
    S_indices = np.where(proxy_scores >= tau)
    Sw_indices = np.arange(len(proxy_scores))  # assuming Sw is the whole set

    numerator = np.sum((oracle_labels[S_indices] == 1) * weights[S_indices])
    denominator = np.sum(oracle_labels[Sw_indices] * weights[Sw_indices])

    return numerator / denominator if denominator != 0 else 0

def PrecisionSw(tau, proxy_scores, oracle_labels, weights):
    S_indices = np.where(proxy_scores >= tau)
    Sw_indices = np.arange(len(proxy_scores))  # assuming Sw is the whole set
    
    numerator = np.sum((oracle_labels[S_indices] == 1) * weights[S_indices])
    denominator = np.sum(weights[Sw_indices])
    
    return numerator / denominator if denominator != 0 else 0

def binary_search_tau(gamma, A, O, m, tol=1e-15):
    low = 0
    high = 1

    while high - low > tol:
        mid = (low + high) / 2
        if RecallSw(mid, A, O, m) >= gamma:
            low = mid
        else:
            high = mid
            
    return low


def tau_IS_CI_R(D, A, O, s, gamma, delta):
    # Compute the weights ~w
    weights = np.array([np.sqrt(A[x]) for x in range(D)])

    # Defensive Mixing: normalize the weights
    weights = 0.9 * weights / np.sum(weights) + 0.1 / D

    # WeightedSample: sample s indices from D based on weights
    S_indices = np.random.choice(range(D), size=s, replace=False, p=weights)

    # Compute m(x)
    m = 1 / (weights * D)

    # Compute tau_o
    tau_o = binary_search_tau(gamma, A, O, m)

    # Compute z^1 and z^2
    z1 = np.array([int(A[i] >= tau_o) * O[i] * m[i] for i in S_indices])
    z2 = np.array([int(A[i] < tau_o) * O[i] * m[i] for i in S_indices])

    # Compute gamma0 using UB and LB functions
    ub = UB(np.mean(z1), np.std(z1), s, delta/2)
    lb =  LB(np.mean(z2), np.std(z2), s, delta/2)
    gamma_new = ub / (ub + lb)

    # Compute tau_0
    tau_new = binary_search_tau(gamma_new, A, O, m)
    print('gamma_o: ', gamma, 'gamma_new: ', gamma_new, 'tau_o: ', tau_o, 'tau_new: ',tau_new)
    return tau_new


def tau_IS_CI_P(D, A, O, s, gamma, delta):
    m = 10  # Minimum step size

    # Compute the weights ~w
    weights = np.array([np.sqrt(A[x]) for x in range(D)])

    # Defensive Mixing: normalize the weights
    weights = 0.9 * weights / np.sum(weights) + 0.1 / D

    # WeightedSample: sample s indices from D based on weights
    S0_indices = np.random.choice(range(D), size=int(s/2), replace=False, p=weights) # stage 1
    # Compute m(x)
    m_x = 1 / (weights * D)

    # Compute Z
    Z = np.array([O[i] * m_x[i] for i in S0_indices])

    # Compute n_match
    n_match = int(D * UB(np.mean(Z), np.std(Z), s/2, delta/2))

    print("n_match:", n_match)
    
    # Sort A in descending order
    A_sorted = sorted([(A[i], i) for i in range(D)], reverse=True)
    D0_indices = np.array([x[1] for x in A_sorted[:int(n_match/gamma)]])  # Get indices of the top n_match scores

    # Stage 2
    weights_D0 = weights[D0_indices]
    weights_D0 = weights_D0 / weights_D0.sum()  # normalize the weights 
    S1_indices = np.random.choice(D0_indices, size=int(s/2), replace=False, p=weights_D0) 

    A_S1 = np.array([A[i] for i in S1_indices])
    s = int(s/2)
    M = int(np.ceil(s/m))

    # Initialize candidates
    candidates = []

    for i in range(m, s, m):
        tau = A_S1[i]
 
        # Compute Z
        Z = np.array([O[j] for j in S1_indices if A[j] >= tau])

        # Compute precision lower bound
        p_l = LB(np.mean(Z), np.std(Z), len(Z), (delta/(2*M)))

        if p_l > gamma:
            candidates.append(tau)
            
    return min(candidates) if candidates else None  # return None if there are no candidates


def supg_query(D, A, O, s, gamma, delta, target_type):
    # Assuming D is a list or array of data points
    # A is a function that takes a data point and returns a score
    # O is a function that takes a data point and returns its oracle label

    S_indices = np.random.choice(range(D), size=s, replace=False)
    
    if (target_type == "RECALL"):
        tau = tau_IS_CI_R(D, A, O, s, gamma, delta)   # Recall target

    if (target_type == "PRECISION"):
        tau = tau_IS_CI_P(D, A, O, s, gamma, delta)  # Precision target
    
    # Creating the sets R1 and R2
    R1 = [x for x in S_indices if O[x] == 1]
    if (tau != None):
        R2 = [x for x in range(D) if A[x] >= tau]
    else:
        R2 = []

    # Return the union of R1 and R2
    return R1 + R2


def evaluate(res, res_accurate):
    res_set = set(res)
    res_accurate_set = set(res_accurate)

    # Calculate precision
    precision = len(res_set.intersection(res_accurate_set)) / len(res_set)

    # Calculate recall
    recall = len(res_set.intersection(res_accurate_set)) / len(res_accurate_set)

    print("Precision: ", precision)
    print("Recall: ", recall)

    return precision, recall


def main():
    A = np.load('proxy_scores_dataset.npy')
    O = np.load('oracle_labels_dataset.npy')
    
    # Example usage
    query = "SELECT * FROM beta WHERE 1 ORACLE LIMIT 1000 USING proxy_estimates [PRECISION] TARGET 0.95 WITH PROBABILITY 0.95"
    query = parse_query(query)
    print(query)

    res = supg_query(n_records, A, O, query['s'], query['target'], query['delta'], query['proxy_extimates'])
    res.sort()
    res_accurate = [i for i in range(n_records) if O[i] == 1]
    evaluate(res, res_accurate) 

    precisions = []
    recalls = []

    for _ in range(20):
        res = supg_query(n_records, A, O, query['s'], query['target'], query['delta'], query['proxy_extimates'])
        res.sort()
        res_accurate = [i for i in range(n_records) if O[i] == 1]
        precision, recall = evaluate(res, res_accurate)
        precisions.append(precision)
        recalls.append(recall)

    x_values = list(range(1,21))

    plt.plot(x_values, precisions)
    plt.xlabel('Trial number')
    plt.ylabel('Precision')
    plt.title('Precision of 20 trials of SUPG with a precision target of 95%')
    plt.xticks(x_values, [str(i) for i in x_values])
    plt.show()

    # plt.plot(x_values, recalls)
    # plt.xlabel('Trial number')
    # plt.ylabel('Precision')
    # plt.title('Precision of 20 trials of SUPG with a recall target of 95%')
    # plt.xticks(x_values, [str(i) for i in x_values])
    # plt.show()
    

if __name__ == "__main__":
    main()
