import numpy as np
from scipy.optimize import linprog

class Auction:
    def __init__(self, *args, **kwargs):
        pass

    def get_winners(self, bids):
        pass

    def get_payments_per_click(self, winners, values, bids):
        pass

    def round(self, bids):
        winners, values = self.get_winners(bids) # allocation mechanism!
        payments_per_click = self.get_payments_per_click(winners, values, bids)
        return winners, payments_per_click
    
class MultiplicativePacingAgent:
    def __init__(self, valuation, budget, T, eta):
        self.valuation = valuation
        self.budget = budget
        self.eta = eta
        self.T = T
        self.rho = self.budget/self.T
        self.lmbd = 1
        self.t = 0

    def bid(self):
        if self.budget < 1:
            return 0
        return self.valuation/(self.lmbd+1)
    
    def update(self, f_t, c_t):
        self.lmbd = np.clip(self.lmbd-self.eta*(self.rho-c_t), 
                            a_min=0, a_max=1/self.rho)
        self.budget -= c_t

class UCBLikeAgent():
    def __init__(self, budget, T, bids):
        self.bids = bids
        self.K = len(self.bids)
        self.budget = budget
        self.T = T
        self.b_t = None
        self.f_avg = np.zeros(self.K)
        self.c_avg = np.zeros(self.K)
        self.N_pulls = np.zeros(self.K)
        self.rho = budget/T
        self.t = 0
    
    def bid(self):
        if self.budget < 1:
            return 0
        elif self.t < self.K:
            self.b_t = self.t # we first try every bid
            return self.b_t
        else:   
            f_ucbs = self.f_avg + np.sqrt(2*np.log(self.T)/self.N_pulls)
            c_ucbs = self.c_avg - np.sqrt(2*np.log(self.T)/self.N_pulls)

            # Finding distribution gamma with linear programming
            c = -f_ucbs  

            A = c_ucbs.reshape(1,self.K) 
            b = np.array([self.rho])     # Upper bound for the inequality constraint

            A_eq = np.ones((1, self.K))  
            b_eq = np.array([1])  # The sum of the distribution should be 1

            # Solving the linear program
            result = linprog(c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=(0,1), method='highs')

            # Extract the optimized distribution over bids
            distribution = result.x.T
            sampled_bid = np.random.choice(np.arange(self.K), p=distribution)
            self.b_t = sampled_bid
            
            return self.bids[self.b_t]
    
    def update(self, f_t, c_t):
        self.N_pulls[self.b_t] += 1
        self.f_avg[self.b_t] += (f_t - self.f_avg[self.b_t])/self.N_pulls[self.b_t]
        self.c_avg[self.b_t] += (c_t - self.c_avg[self.b_t])/self.N_pulls[self.b_t]
        
        self.budget -= c_t
        self.t += 1

# Agent EXP3 with Primal-Dual adjustments
class EXP3AgentPrimalDual:
    def __init__(self, num_slots, num_prices, learning_rate):
        self.num_slots = num_slots
        self.num_prices = num_prices
        self.learning_rate = learning_rate
        
        # Initialize weights for both bids and prices
        self.bid_weights = np.ones(num_slots)
        self.price_weights = np.ones(num_prices)
        
        self.bid_probabilities = self.bid_weights / self.bid_weights.sum()
        self.price_probabilities = self.price_weights / self.price_weights.sum()

    def bid(self):
        self.bid_probabilities = (1 - self.learning_rate) * (self.bid_weights / self.bid_weights.sum()) + \
                                 (self.learning_rate / self.num_slots)
        bid_slot = np.random.choice(self.num_slots, p=self.bid_probabilities)
        return bid_slot

    def choose_price(self):
        self.price_probabilities = (1 - self.learning_rate) * (self.price_weights / self.price_weights.sum()) + \
                                   (self.learning_rate / self.num_prices)
        price_level = np.random.choice(self.num_prices, p=self.price_probabilities)
        return price_level

    def update(self, f_t, c_t):
        bid_slot = self.last_bid_slot  # Assume we store the last bid slot
        price_level = self.last_price_level  # Assume we store the last price level

        # Update bid weights
        estimated_loss = c_t / self.bid_probabilities[bid_slot]
        self.bid_weights[bid_slot] *= np.exp(-self.learning_rate * estimated_loss / self.num_slots)

        # Update price weights
        estimated_reward = f_t / self.price_probabilities[price_level]
        self.price_weights[price_level] *= np.exp(self.learning_rate * estimated_reward / self.num_prices)

        # Update probabilities
        self.bid_probabilities = (1 - self.learning_rate) * (self.bid_weights / self.bid_weights.sum()) + \
                                 (self.learning_rate / self.num_slots)
        self.price_probabilities = (1 - self.learning_rate) * (self.price_weights / self.price_weights.sum()) + \
                                   (self.learning_rate / self.num_prices)

    def update_price(self, price_level, reward):
        estimated_reward = reward / self.price_probabilities[price_level]
        self.price_weights[price_level] *= np.exp(self.learning_rate * estimated_reward / self.num_prices)