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

class EXP3AgentPrimalDual:
    def __init__(self, possible_bids, learning_rate, budget, eta, T):
        self.possible_bids = possible_bids

        self.learning_rate = learning_rate
        self.budget = budget
        self.remaining_budget = self.budget
        self.eta = eta
        self.rho = self.budget/T
        
        # Initialize weights for both bids and prices
        self.bid_weights = np.ones(len(possible_bids))
        
        self.bid_probabilities = self.bid_weights / self.bid_weights.sum()
        
        # Initialize lambda (pacing multiplier)
        self.lmbd = 1.0
        self.current_day = 0
        self.T = T

        self.last_prob = None

    def bid(self):
        if self.current_day >= self.T:
            return 0
        
        # Calculate remaining per-round budget
        remaining_rounds = self.T - self.current_day
        per_round_budget = self.remaining_budget / remaining_rounds

        # Add a small constant to prevent division by zero
        epsilon = 1e-10
        probs = (self.bid_weights + epsilon) / np.sum(self.bid_weights + epsilon)
        
        # Ensure probabilities sum to 1
        probs /= np.sum(probs)

        self.last_prob = probs
        chosen_bid_index = np.random.choice(len(self.possible_bids), p=probs)
        chosen_bid = self.possible_bids[chosen_bid_index]
        
        returned_bid = min(chosen_bid, per_round_budget)

        return returned_bid

    def update(self, utility, cost):
        # Update weights
        loss = self.lmbd * (cost - self.rho) - utility
        estimated_loss = loss / (self.last_prob + 1e-10)  # Add small constant to avoid division by zero
        
        # Clip the estimated loss to prevent extreme values
        clipped_loss = np.clip(estimated_loss, -10, 10)
        
        # Update weights using softmax to maintain numerical stability
        self.bid_weights = np.exp(np.log(self.bid_weights) - self.eta * clipped_loss)
        self.bid_weights /= np.sum(self.bid_weights)  

        # Update lambda
        self.lmbd = max(0, self.lmbd + self.eta * (cost - self.rho))

        self.remaining_budget -= cost
        self.current_day += 1