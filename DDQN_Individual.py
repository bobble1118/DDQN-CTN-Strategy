# Description: 使用 Double Deep Q-Learning Network (DDQN) 訓練 Maze 環境的智能體，並測試其效能
import torch
import matplotlib.pyplot as plt
import Environment as env
import Agent 

def plot_result(rewards_history, loss_history):
    plt.figure(figsize=(12, 5))

    # First plot: step
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(rewards_history) + 1), rewards_history, label="Rewards per Episode", color='royalblue')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")
    plt.legend()

    # Second plot: reward
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(loss_history) + 1), loss_history, label="Loss per Episode", color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Loss per Episode")
    plt.legend()

    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    # 設定運算設備：有 GPU 就用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    # 建立 Maze 環境
    Maze = env.MazeEnv()
    start_position = Maze.start_position                  # 狀態空間：[x, y]
    actions = Maze.actions                                # 動作數量：4
    agent = Agent.DDQNAgent(state=start_position, actions=actions, learning_rate=0.01, gamma=0.9, 
                      buffer_capacity=100, batch_size=10, device=device)
    
    num_episodes = 400  # 總訓練回合數
    max_steps = 100     # 每回合最多執行步數        
    rewards_history = []
    loss_history = []
    
    # 訓練迴圈
    for episode in range(num_episodes):
        agent.state = list(Maze.reset())
        total_reward = 0
        for step in range(max_steps):
            # 利用 epsilon-greedy 策略選擇動作
            action_index = agent.choose_action(agent.state)

            # 將動作索引轉換成 MazeEnv 所定義的字串動作
            action_str = Maze.actions[action_index]
            
            next_state, reward, done = Maze.take_action(action_str, agent.state)
            total_reward += reward
            # print(f"Step {step+1}: {agent.state} --{action_str}--> {next_state}, reward: {reward}")

            # 儲存經驗至重放記憶體
            agent.store_transition(agent.state, action_index, reward, next_state, done)
            
            # 每個 step 執行一次訓練更新
            loss_value = agent.train()
        
            agent.state = next_state
            
            if done:
                print(f"---------Episode {episode+1} done in {step+1} steps----------")  
                break
        
        rewards_history.append(total_reward)
        loss_history.append(loss_value)
        print(f"Episode {episode+1}: total reward = {total_reward:.3f}, epsilon = {agent.epsilon:.3f}, loss = {loss_value:.3f}")
    
    # 顯示訓練結果
    plot_result(rewards_history, loss_history)
    
    # 測試已訓練好的 Agent
    agent.state = Maze.reset()
    done = False
    print("開始測試 Agent...")
    for step in range(max_steps):
        # 渲染當前環境
        action_index = agent.choose_action(agent.state)
        action_str = agent.actions[action_index]
        agent.state, reward, done = Maze.take_action(action_str, agent.state)
        if done:
            print("到達目標！")
            break
    
    # 最後再顯示一次環境
    Maze.render([agent.state])

        
    
