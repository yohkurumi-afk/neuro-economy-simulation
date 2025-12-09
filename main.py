import numpy as np
import matplotlib.pyplot as plt

class NeuroAgent:
    def __init__(self, id, risk_profile='neutral'):
        self.id = id
        self.cash = 1000  # 初期資産
        self.stock = 0
        self.risk_profile = risk_profile
        # Q-table: 状態(価格が上がった/下がった) x 行動(買い/売り/保持)
        self.q_table = np.zeros((2, 3)) 
        
    def act(self, current_price_trend):
        # ここに「行動選択（イプシロン・グリーディ法など）」を実装
        # 0: Buy, 1: Sell, 2: Hold
        return np.random.choice([0, 1, 2]) 

    def learn(self, reward):
        # ここに「強化学習の更新式」を実装
        # risk_profileによって学習率などを変える
        pass

class Market:
    def __init__(self):
        self.price = 100
        self.history = []
        
    def step(self, actions):
        # 買い注文と売り注文のバランスで価格を更新
        buy_orders = actions.count(0)
        sell_orders = actions.count(1)
        
        # 単純な需要供給モデル
        if buy_orders > sell_orders:
            self.price *= 1.01
        elif sell_orders > buy_orders:
            self.price *= 0.99
            
        self.history.append(self.price)
        return self.price

# --- Simulation Execution ---
if __name__ == "__main__":
    market = Market()
    agents = [NeuroAgent(i) for i in range(50)] # 50人のエージェント

    print("Simulation Start...")
    for t in range(100):
        actions = [agent.act(0) for agent in agents]
        new_price = market.step(actions)
    
    print("Simulation Complete. Showing graph...")
    # 結果のプロット
    plt.plot(market.history)
    plt.title("Simulated Market Price (Random Agents)")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.show()