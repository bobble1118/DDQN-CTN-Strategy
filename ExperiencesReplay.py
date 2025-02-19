import numpy as np
import random
from collections import deque

# 重放記憶體：儲存 Agent 的經驗，方便後續隨機取樣進行訓練
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # 將 batch 中的各個項目分解成獨立的陣列
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)