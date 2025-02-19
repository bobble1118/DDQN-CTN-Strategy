import torch
import torch.nn as nn
import torch.optim as optim
import random

import Network as Net
import ExperiencesReplay as Exp

# DDQN Agent
class DDQNAgent:
    def __init__(self, state, actions,learning_rate=0.01, gamma=0.9, 
                 buffer_capacity=10000, batch_size=64, device='cpu'):
       # Agent Information
        self.state = state
        self.actions = actions

        # Hyperparameters   
        self.state_size = len(state)
        self.action_size = len(actions)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        # Experience Replay
        self.replay_buffer = Exp.ReplayBuffer(buffer_capacity)
        
        # Build Online Network and Target Network
        self.online_network = Net.QNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network = Net.QNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
        
        # 將 target 網路初始權重與線上網路同步
        self.update_target_network()
        
        # Epsilon 用於 epsilon-greedy 探索策略
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.99988
        
        self.train_step = 0
        self.update_target_every = 5  # 每5次訓練後更新 target 網路

    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def choose_action(self, state):
        # 依照 epsilon-greedy 策略選擇動作
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # shape: [1, self.state_size]
            with torch.no_grad():
                q_values = self.online_network(state_tensor)
            return q_values.argmax().item()
        
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train(self):
        # 當記憶體中的經驗不足一個 batch 時，不進行訓練
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 將 numpy 陣列轉換成 tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 取得目前 Q 值：對於每個 state，選取對應所選動作的 Q 值
        q_values = self.online_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # DDQN 更新：
        # 1. 使用線上網路選出在下一個 state 下最佳的動作
        # 2. 再利用 target 網路評估該動作的 Q 值
        with torch.no_grad():
            next_q_values = self.online_network(next_states)
            next_actions = next_q_values.argmax(dim=1)
            next_q_target_values = self.target_network(next_states)
            next_q_value = next_q_target_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # 計算 target Q 值
            expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)
        
        # 損失函數：均方誤差
        loss = nn.MSELoss()(q_value, expected_q_value)
        
        # 反向傳播更新參數
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.train_step += 1

        # 每隔一定步數，更新 target 網路參數
        if self.train_step % self.update_target_every == 0:
            self.update_target_network()
            
        # 更新 epsilon，逐漸降低探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
