from QRobot import QRobot

class Robot(QRobot):

    def __init__(self, maze):
        """
        初始化 Robot 类
        :param maze:迷宫对象
        """
        super(Robot, self).__init__(maze)
        self.maze = maze

    def train_update(self):
        """
        以训练状态选择动作并更新Deep Q network的相关参数
        :return :action, reward 如："u", -1
        """
        action, reward = "u", -1.0

        # -----------------请实现你的算法代码--------------------------------------

        # -----------------------------------------------------------------------

        return action, reward

    def test_update(self):
        """
        以测试状态选择动作并更新Deep Q network的相关参数
        :return : action, reward 如："u", -1
        """
        action, reward = "u", -1.0

        # -----------------请实现你的算法代码--------------------------------------

        # -----------------------------------------------------------------------

        return action, reward

    from QRobot import QRobot
    from Maze import Maze

    maze = Maze(maze_size=5)  # 随机生成迷宫

    robot = QRobot(maze)  # 记得将 maze 变量修改为你创建迷宫的变量名

    action, reward = robot.train_update()  # QLearning 算法一次Q值迭代和动作选择

    print("the choosed action: ", action)
    print("the returned reward: ", action)

    from QRobot import QRobot
    from Maze import Maze
    from Runner import Runner

    """  Qlearning 算法相关参数： """

    epoch = 10  # 训练轮数
    epsilon0 = 0.5  # 初始探索概率
    alpha = 0.5  # 公式中的 ⍺
    gamma = 0.9  # 公式中的 γ
    maze_size = 5  # 迷宫size

    """ 使用 QLearning 算法训练过程 """

    g = Maze(maze_size=maze_size)
    r = QRobot(g, alpha=alpha, epsilon0=epsilon0, gamma=gamma)

    runner = Runner(r)
    runner.run_training(epoch, training_per_epoch=int(maze_size * maze_size * 1.5))

    # 生成训练过程的gif图, 建议下载到本地查看；也可以注释该行代码，加快运行速度。
    runner.generate_gif(filename="results/size5.gif")





    """ReplayDataSet 类的使用"""

    from ReplayDataSet import ReplayDataSet

    test_memory = ReplayDataSet(max_size=1e3)  # 初始化并设定最大容量
    actions = ['u', 'r', 'd', 'l']
    test_memory.add((0, 1), actions.index("r"), -10, (0, 1), 1)  # 添加一条数据（state, action_index, reward, next_state）
    print(test_memory.random_sample(1))  # 从中随机抽取一条（因为只有一条数据）




