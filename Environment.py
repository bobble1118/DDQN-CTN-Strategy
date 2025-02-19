import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

# Environment for Maze 
class MazeEnv:
    def __init__(self, start_position = [0.5, 0.5], goal_position = [7.5, 7.5]):
        self.grid_size = 7
        self.walls = [       # (x, y, w, h)
            (2, 0, 1, 3),    
            (0, 5, 3, 1),    
            (5, 3, 1, 3),   
        ]
        self.start_position = start_position
        self.goal_position = goal_position
        self.object_position = [1, 0.5]

        # ACTIONS
        self.actions = ['forward', 'backward', 'left', 'right']

    def reset(self):
        # Reset the environment
        return self.start_position

    # def get_state(self, agent_positions):
    #     # 獲取當前狀態
    #     return [agent_positions[0], agent_positions[1]]
    
    def meetWall(self, agent_positions):
        # 判斷是否碰到牆壁
        for wall in self.walls:
            x, y, w, h = wall
            if agent_positions[0] >= x and agent_positions[0] <= x + w and agent_positions[1] >= y and agent_positions[1] <= y + h:
                return True
        if agent_positions[0] < 0 or agent_positions[0] >= self.grid_size + 1 or agent_positions[1] < 0 or agent_positions[1] >= self.grid_size + 1:
            return True
        return False

    def take_action(self, action, state):
        agent_positions = copy.deepcopy(state)
        current_state = list(agent_positions)
        meet_wall = False

        # 執行動作並更新狀態
        if action == 'forward':
            agent_positions[1] += 1
        elif action == 'backward':
            agent_positions[1] -= 1
        elif action == 'left':
            agent_positions[0] -= 1
        elif action == 'right':
            agent_positions[0] += 1

        # 遇到牆壁返回
        if self.meetWall(agent_positions):
            agent_positions[:] = current_state  
            meet_wall = True

        # 計算獎勵
        reward, done = self.compute_reward(agent_positions, meet_wall)

        return agent_positions, reward, done


    def compute_reward(self, agent_positions, meet_wall=False):
        # 計算獎勵和是否結束
        if (int(agent_positions[0]), int(agent_positions[1])) == (7, 7):
            return 100, True  # 到達目標
        if meet_wall:
            return -10, False  # 碰壁失敗
        else:   
            return -0.1, False  # 每步給負獎勵以促使更快完成目標

    def render(self, agent_positions, enable_object=False):
        # 使用 matplotlib 繪製環境
        fig, ax = plt.subplots(figsize=(8, 8))

        # 畫出牆壁
        for wall in self.walls:
            x, y, w, h = wall
            ax.add_patch(patches.Rectangle((x, y), w, h, color="black", zorder=1))

        # 畫出障礙物標籤
        obstacles = [
            (2.5, 1.4),  
            (1.5, 5.4),  
            (5.5, 4.4),  
        ]
        for obs in obstacles:
            ax.text(*obs, "Obstacles", fontsize=10, ha="center", color='white', zorder=2)

        # 畫出起始點 (S)
        ax.add_patch(patches.Circle(self.start_position, 0.4, color="blue", label="Starting Position", zorder=1))
        ax.text(*self.start_position, "S", color="white", fontsize=12, ha="center", va="center", zorder=2)

        # 畫出目標點 (G)
        ax.add_patch(patches.Rectangle((7, 7), 1, 1, color="red", label="Goal", zorder=1))
        ax.text(*self.goal_position, "G", color="white", fontsize=12, ha="center", va="center", zorder=2)

        if enable_object:
            # 畫出物體
            ax.add_patch(patches.Rectangle(
                (self.object_position[0] - 0.5, self.object_position[1] - 0.25), 1, 0.5, 
                color="orange", label="Object", zorder=3
            ))
            ax.text(self.object_position[0], self.object_position[1]-0.05, "Object", fontsize=8, ha="center", zorder=4)

        # 畫出兩個 Agent，並確保它們在最上層
        for i, agent_pos in enumerate(agent_positions):
            ax.add_patch(patches.Circle(agent_pos, 0.3, color="green", label=f"Agent {i+1}", zorder=4))
            ax.text(*agent_pos, str(i+1), color="white", fontsize=10, ha="center", va="center", zorder=5)

        # 設定座標範圍與網格
        ax.set_xlim(0, self.grid_size + 1)
        ax.set_ylim(0, self.grid_size + 1)
        ax.set_xticks(range(self.grid_size + 2))
        ax.set_yticks(range(self.grid_size + 2))
        ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, zorder=1)
        ax.set_aspect('equal', adjustable='box')

        plt.show()
