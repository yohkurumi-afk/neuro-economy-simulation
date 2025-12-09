import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# --- 設定パラメータ ---
NUM_AGENTS = 50       # エージェントの数
SIMULATION_STEPS = 1000 # シミュレーションの期間
INITIAL_PRICE = 100   # 株価の初期値

class NeuroAgent:
    def __init__(self, id, risk_profile='neutral'):
        self.id = id
        self.cash = 1000.0
        self.stock = 0
        self.risk_profile = risk_profile
        
        # プロスペクト理論パラメータ: 損失をどれくらい嫌がるか
        # 一般的に人間は「2.25倍」損失を嫌うと言われています
        self.loss_aversion_ratio = 2.25 
        
        # Q-table: 状態(2種類) x 行動(3種類)
        self.q_table = np.zeros((2, 3)) 
        
        # 学習用
        self.last_action = 2 
        self.last_state = 0
        self.last_asset = self.get_total_asset(INITIAL_PRICE)

    def get_total_asset(self, current_price):
        """現在の総資産（現金 + 株式評価額）を計算"""
        return self.cash + self.stock * current_price

    def act(self, current_price_trend, epsilon=0.1):
        """行動を選択する (ε-greedy法)"""
        state_idx = int(current_price_trend > 0)
        self.last_state = state_idx

        if np.random.random() < epsilon:
            action = np.random.choice([0, 1, 2])
        else:
            action = np.argmax(self.q_table[state_idx])
        
        self.last_action = action
        return action

    def learn(self, current_price, current_price_trend):
        """結果を見て脳（Q-table）を更新する"""
        alpha = 0.1
        gamma = 0.9
        
        # 資産の変化額（物理的な損益）
        current_asset = self.get_total_asset(current_price)
        asset_change = current_asset - self.last_asset
        self.last_asset = current_asset

        # --- ★ここがプロスペクト理論の実装箇所 ---
        # 心理的な報酬（Subjective Reward）に変換する
        if asset_change < 0:
            # 損失の場合、痛みを増幅させる（例: -100円 -> -225の痛み）
            reward = asset_change * self.loss_aversion_ratio
        else:
            # 利益の場合はそのまま（例: +100円 -> +100の喜び）
            reward = asset_change
        # ---------------------------------------

        # Q学習の更新式
        next_state_idx = int(current_price_trend > 0)
        max_future_q = np.max(self.q_table[next_state_idx])
        current_q = self.q_table[self.last_state][self.last_action]
        
        td_error = reward + gamma * max_future_q - current_q
        self.q_table[self.last_state][self.last_action] += alpha * td_error

class Market:
    def __init__(self):
        self.price = INITIAL_PRICE
        self.history = [self.price]
        self.trend = 0
        
    def step(self, actions):
        """価格更新メカニズム"""
        buy_orders = actions.count(0)
        sell_orders = actions.count(1)
        
        # 需給による価格変動
        if buy_orders > sell_orders:
            self.price *= (1 + 0.005 * (buy_orders - sell_orders))
            self.trend = 1
        elif sell_orders > buy_orders:
            self.price *= (1 - 0.005 * (sell_orders - buy_orders))
            self.trend = -1
        else:
            self.trend = 0

        # 市場のゆらぎ（ノイズ(N（0, 0.02））
        noise = np.random.normal(0, 0.02)
        self.price *= (1 + noise)

        self.history.append(self.price)
        return self.price, self.trend

# --- メイン実行部分 ---
if __name__ == "__main__":
    # 結果保存フォルダの作成
    if not os.path.exists('results'):
        os.makedirs('results')

    market = Market()
    agents = [NeuroAgent(i) for i in range(NUM_AGENTS)]
    
    # 記録用データ
    avg_assets_history = []

    print("Simulation Start...")
    
    for t in range(SIMULATION_STEPS):
        # 1. 市場トレンド取得
        if len(market.history) < 2:
            trend = 0
        else:
            trend = market.history[-1] - market.history[-2]

        # 2. 全エージェントの行動
        actions = []
        for agent in agents:
            action = agent.act(trend)
            actions.append(action)
            
            # 売買実行
            if action == 0 and agent.cash >= market.price: # Buy
                agent.stock += 1
                agent.cash -= market.price
            elif action == 1 and agent.stock > 0: # Sell
                agent.stock -= 1
                agent.cash += market.price

        # 3. 市場更新
        current_price, _ = market.step(actions)

        # 4. 学習 & データ記録
        total_assets = 0
        for agent in agents:
            agent.learn(current_price, trend)
            total_assets += agent.get_total_asset(current_price)
        
        avg_assets_history.append(total_assets / NUM_AGENTS)

    print("Simulation Complete.")

    # --- グラフの描画と保存 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 上段：市場価格の推移
    ax1.plot(market.history, color='blue')
    ax1.set_title("Market Price Trend")
    ax1.set_ylabel("Price")
    
    # 下段：平均資産の推移（ここが右肩上がりなら学習成功！）
    ax2.plot(avg_assets_history, color='green')
    ax2.set_title("Average Agent Asset (Learning Progress)")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Average Asset")
    
    plt.tight_layout()
    
    # 画像ファイルとして保存 (ファイル名に日時をつける)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/sim_{timestamp}.png"
    plt.savefig(filename)
    print(f"Graph saved to: {filename}")
    
    # 画面にも表示
    plt.show()