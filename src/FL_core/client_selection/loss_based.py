from copy import deepcopy
from collections import Counter
import numpy as np

from .client_selection import ClientSelection



'''Active Federated Learning'''
class ActiveFederatedLearning(ClientSelection):
    def __init__(self, total, device, args):
        super().__init__(total, device)
        self.alpha1 = args.alpha1 #0.75
        self.alpha2 = args.alpha2 #0.01
        self.alpha3 = args.alpha3 #0.1
        self.save_probs = args.save_probs

    def select(self, n, client_idxs, metric, round=0, results=None):
        # set sampling distribution
        values = np.exp(np.array(metric) * self.alpha2)
        # 1) select 75% of K(total) users
        num_drop = len(metric) - int(self.alpha1 * len(metric))
        drop_client_idxs = np.argsort(metric)[:num_drop]
        probs = deepcopy(values)
        probs[drop_client_idxs] = 0
        probs /= sum(probs)
        #probs = np.nan_to_num(probs, nan=max(probs))
        # 2) select 99% of m users using prob.
        num_select = int((1 - self.alpha3) * n)
        #np.random.seed(round)
        selected = np.random.choice(len(metric), num_select, p=probs, replace=False)
        # 3) select 1% of m users randomly
        not_selected = np.array(list(set(np.arange(len(metric))) - set(selected)))
        selected2 = np.random.choice(not_selected, n - num_select, replace=False)
        selected_client_idxs = np.append(selected, selected2, axis=0)
        print(f'{len(selected_client_idxs)} selected users: {selected_client_idxs}')

        if self.save_probs:
            self.save_results(metric, results, f'{round},loss,')
            self.save_results(values, results, f'{round},value,')
            self.save_results(probs, results, f'{round},prob,')
        return selected_client_idxs.astype(int)



'''Power-of-Choice'''
class PowerOfChoice(ClientSelection):
    def __init__(self, total, device, d):
        super().__init__(total, device)
        #self.d = d

    def setup(self, n_samples):
        client_ids = sorted(n_samples.keys())
        n_samples = np.array([n_samples[i] for i in client_ids])
        self.weights = n_samples / np.sum(n_samples)

    def select_candidates(self, client_idxs, d):
        # 1) sample the candidate client set
        weights = np.take(self.weights, client_idxs)
        candidate_clients = np.random.choice(client_idxs, d, p=weights/sum(weights), replace=False)
        return candidate_clients

    def select(self, n, client_idxs, metric, round=0, results=None):
        # 3) select highest loss clients
        selected_client_idxs = np.argsort(metric)[-n:]
        return selected_client_idxs




'''
import random

class MaxRate(ClientSelection):
    def __init__(self, total, device,bandwidth_limit, power_limit, num_clients):
        super().__init__(total, device)
        self.bandwidth_limit = bandwidth_limit
        self.power_limit = power_limit
        self.num_clients = num_clients

    def setup(self, n_samples):
        client_ids = sorted(n_samples.keys())
        n_samples = np.array([n_samples[i] for i in client_ids])
        self.weights = n_samples / np.sum(n_samples)
    
    def init(self, global_m, local_models):
        self.prev_global_m = global_m
        self.gradients = self.get_gradients(global_m, local_models)

    # def select_candidates(self, client_idxs, d):
    #     # 1) sample the candidate client set
    #     weights = np.take(self.weights, client_idxs)
    #     candidate_clients = np.random.choice(client_idxs, d, p=weights/sum(weights), replace=False)
    #     return candidate_clients

    def select_candidates(self, client_idxs, d):
        # 1) sample the candidate client set
        weights = np.take(self.weights, client_idxs)
        candidate_clients = np.random.choice(client_idxs, d, p=weights/sum(weights), replace=False)
        return candidate_clients

    def get_gradients(self, global_m, local_models):
        """
        return the `representative gradient` formed by the difference
        between the local work and the sent global model
        """
        local_model_params = []
        for model in local_models:
            local_model_params += [[tens.detach().to(self.device) for tens in list(model.parameters())]] #.numpy()

        global_model_params = [tens.detach().to(self.device) for tens in list(global_m.parameters())]

        local_model_grads = []
        for local_params in local_model_params:
            local_model_grads += [[local_weights - global_weights
                                   for local_weights, global_weights in
                                   zip(local_params, global_model_params)]]

        return local_model_grads


    def select_clients(self, num_selected):
        """
        Select top N clients based on randomly generated SNR values, and meet the bandwidth and power limit.

        Args:
        - num_selected: number of clients to select

        Returns:
        - a list of indices of selected clients
        """
        # Generate random SNR values for each client
        snr_values = [random.uniform(0, 1) for i in range(self.num_clients)]

        # Sort the clients by their SNR values in descending order
        sorted_clients = sorted(range(len(snr_values)), key=lambda x: snr_values[x], reverse=True)

        selected_client_idxs = []
        total_bandwidth = 0
        total_power = 0

        # Select top N clients that meet the bandwidth and power limit
        for i in range(num_selected):
            client = sorted_clients[i]

            # Check if the bandwidth and power limit will be exceeded
            if total_bandwidth + self.bandwidth_limit[client] <= self.power_limit[client] and \
               total_power + self.power_limit[client] <= self.power_limit:
                selected_clients.append(client)
                total_bandwidth += self.bandwidth_limit[client]
                total_power += self.power_limit[client]

        return selected_client_idxs.astype(int)


    # def select_clients(self, snr_values, num_selected):
    #     """
    #     Select top N clients based on SNR ranking, and meet the bandwidth and power limit.

    #     Args:
    #     - snr_values: a list of SNR values for each client
    #     - num_selected: number of clients to select

    #     Returns:
    #     - a list of indices of selected clients
    #     """
    #     # Sort the clients by their SNR values in descending order
    #     sorted_clients = sorted(range(len(snr_values)), key=lambda x: snr_values[x], reverse=True)

    #     selected_client_idxs = []
    #     total_bandwidth = 0
    #     total_power = 0

    #     # Select top N clients that meet the bandwidth and power limit
    #     for i in range(num_selected):
    #         client = sorted_clients[i]

    #         # Check if the bandwidth and power limit will be exceeded
    #         if total_bandwidth + self.bandwidth_limit[client] <= self.power_limit[client] and \
    #            total_power + self.power_limit[client] <= self.power_limit:
    #             selected_client_idxs.append(client)
    #             total_bandwidth += self.bandwidth_limit[client]
    #             total_power += self.power_limit[client]

    #     return selected_client_idxs

'''



# class MaxRate(ClientSelection):
#     # import random
#     def __init__(self,total,device,d):
#         super().__init__(total, device)
# # Define total number of clients, total bandwidth and power constraints
# # num_clients = 3000
# # B = 1000
# # P = 100
# # Generate a list of clients with random values for ID, bandwidth, power, and SNR
#     client_list = []
#     for i in range(num_clients):
#         client = {
#             'id': i,
#             'bandwidth': random.randint(1, B),
#             'power': random.randint(1, P),
#             'snr': random.uniform(1, 10)
#         }
#         client_list.append(client)

#     # Sort the client list by SNR in descending order
#     client_list.sort(key=lambda x: x['snr'], reverse=True)

#     # Initialize a list to store the selected clients
#     selected_client_idxs = []

#     # Initialize variables to store the total bandwidth and power
#     total_bandwidth = 0
#     total_power = 0

#     # Iterate through the client list
#     for client in client_list:
#         # Check if the client satisfies the bandwidth and power constraints
#         if total_bandwidth + client['bandwidth'] <= B and total_power + client['power'] <= P:
#             # Add the client to the list of selected clients
#             selected_client_idxs.append(client)
#             total_bandwidth += client['bandwidth']
#             total_power += client['power']
#         if len(selected_client_idxs) == 10:
#             break

#     # Compute the total SNR of the selected clients
#     total_snr = sum([client['snr'] for client in selected_client_idxs])
#     return selected_client_idxs

#     # print("Selected clients:", selected_client_idxs)
#     # print("Total SNR:", total_snr)
