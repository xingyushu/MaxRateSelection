import numpy as np
import random
from utils import utils
from utils.argparse import get_args


class ClientSelection:
    def __init__(self, total, device):
        self.total = total
        self.device = device

    def select(self, n, client_idxs, metric):
        pass

    def save_selected_clients(self, client_idxs, results):
        tmp = np.zeros(self.total)
        tmp[client_idxs] = 1
        tmp.tofile(results, sep=',')
        results.write("\n")

    def save_results(self, arr, results, prefix=''):
        results.write(prefix)
        np.array(arr).astype(np.float32).tofile(results, sep=',')
        results.write("\n")



'''Random Selection'''
class RandomSelection(ClientSelection):
    def __init__(self, total, device):
        super().__init__(total, device)

    def select(self, n, client_idxs, metric=None):
        selected_client_idxs = np.random.choice(client_idxs, size=n, replace=False)
        return selected_client_idxs


"Max-rate"

'''
    parser.add_argument('-A', '--num_clients_per_round', type=int, default=10, help='number of participated clients')
    parser.add_argument('-K', '--total_num_clients', type=int, default=None, help='total number of clients')
'''

args = get_args()


class MaxRate(ClientSelection):
    def __init__(self, total, device, power_limit, bandwidth_limit):
        super().__init__(total, device)
        # self.power_limit = power_limit
        # self.bandwidth_limit = bandwidth_limit

    def select(self, n, client_idxs, metric):


        # Define total number of clients, total bandwidth and power constraints
        # num_clients = 3000
        B = args.power_limit
        P = args.bandwidth_limit
        n = args.total_num_clients
        clients = args.num_clients_per_round

        # Generate a list of clients with random values for ID, bandwidth, power, and SNR
        client_list = []
        for i in range(n):
            client = {
                'id': i,
                'bandwidth': random.randint(1, 10),
                'power': random.randint(1, 10),
                'snr': random.uniform(1, 30)
            }
            client_list.append(client)

        # Sort the client list by SNR in descending order
        client_list.sort(key=lambda x: x['snr'], reverse=True)

        # Initialize a list to store the selected clients
        selected_clients = []

        client_idxs=[]

        # Initialize variables to store the total bandwidth and power
        total_bandwidth = 0
        total_power = 0

        # Iterate through the client list
        for client in client_list:
            # Check if the client satisfies the bandwidth and power constraints
            if total_bandwidth + client['bandwidth'] <= B and total_power + client['power'] <= P:
                # Add the client to the list of selected clients
                selected_clients.append(client)
                total_bandwidth += client['bandwidth']
                total_power += client['power']
            if len(selected_clients) == 10:
                break

        # Compute the total SNR of the selected clients
        for client in selected_clients:
            client_idxs.append(client['id'])
        # print(client_idxs)
        
        return client_idxs

        # total_snr = sum([client['snr'] for client in selected_clients])

        # print("Selected clients:", selected_clients)
        # print("Total SNR:", total_snr)

#     def select(self, n, client_idxs, metric):
#         if metric is None:
#             return np.array([])
#         snr_vals = metric["snr"]
#         power_vals = np.random.rand(len(client_idxs)) * self.power_limit
#         bandwidth_vals = np.random.rand(len(client_idxs)) * self.bandwidth_limit
#         snr_vals = np.random.rand(30)
#         metric["snr"] = snr_vals
#         selected_client_idxs = []
#         for i in client_idxs:
#             if power_vals[i] <= self.power_limit and bandwidth_vals[i] <= self.bandwidth_limit:
#                 selected_client_idxs.append(i)
#         if len(selected_client_idxs) == 0:
#             return np.array([])
#         snr_vals = snr_vals[selected_client_idxs]
#         client_idxs = np.array(selected_client_idxs)
#         top_n_idxs = np.argsort(snr_vals)[-n:]
#         return client_idxs[top_n_idxs]

    # def select(self, n, client_idxs, metric):
    #     if metric is None:
    #         return np.array([])
    #     snr_vals = metric["snr"]
    #     power_vals = metric["power"]
    #     bandwidth_vals = metric["bandwidth"]
    #     snr_vals = np.random.rand(30)
    #     metric["snr"] = snr_vals
    #     selected_client_idxs = []
    #     for i in client_idxs:
    #         if power_vals[i] <= self.power_limit and bandwidth_vals[i] <= self.bandwidth_limit:
    #             selected_client_idxs.append(i)
    #     if len(selected_client_idxs) == 0:
    #         return np.array([])
    #     snr_vals = snr_vals[selected_client_idxs]
    #     client_idxs = np.array(selected_client_idxs)
    #     top_n_idxs = np.argsort(snr_vals)[-n:]
    #     return client_idxs[top_n_idxs]
