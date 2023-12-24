from DQN.maze_env import  Maze
from DQNR.RL_brain import  DeepQNetwork


def update():
    step = 0
    for episode in range(300):
        # 获取初始位置坐标信息（1，1）
        obervation = env.reset()

        while True:
            # 刷新环境
            env.render()

            # RL 选择基于规则的下一个行为
            action = RL.choose_action(obervation)

            # RL 根据采取的行为获取下一个观测和当前的奖励
            obervation_, reward, done = env.step(action)

            # RL 更新replay buffer
            RL.store_transition(obervation,action, reward, obervation_)

            # 首先要有些记忆，所以迭代了一段时间后开始学习，在每5步学习一下
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            obervation = obervation_

            if done:
                break
            step += 1
    print("end of game")
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = DeepQNetwork(
        env.n_actions,env.n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=200,
        memory_size=2000,
        #output_graph=True
    )
env.after(100,update)
env.mainloop()
RL.plot_cost()

# from DQN.maze_env import Maze  # 从DQN目录下的maze_env模块导入Maze类，这是定义的环境或任务
# from DQNR.RL_brain import DeepQNetwork  # 从DQNR目录下的RL_brain模块导入DeepQNetwork类，这是用于解决此任务的强化学习算法
#
#
# def update():  # 定义一个名为update的函数，用于更新环境以及DQN网络的学习
#     step = 0  # 初始化步数，用于记录DQN网络与环境的交互步数
#     for episode in range(300):  # 循环300个episodes，每个episode代表一个完整的任务过程
#         # 重置环境，获取初始观测（状态），此处观测是从(1,1)的位置开始
#         obervation = env.reset()
#
#         while True:  # 在一个任务过程中，会持续与环境交互，直到任务结束
#             # 刷新环境，将环境展示给DQN网络，此处的env.render()可能是一个显示环境的函数，但在给定的代码中并未给出env的定义和render的具体实现
#
#             # DQN网络选择下一个动作，基于当前的观测
#             action = RL.choose_action(obervation)  # 选择一个动作
#
#             # 与环境交互，获取下一个观测、奖励和任务是否结束的信息
#             obervation_, reward, done = env.step(action)  # env.step是一个函数，根据选定的动作执行环境的一步，返回下一个观测、奖励和是否结束的信息
#
#             # 将这一步的信息存储到DQN网络的经验回放缓冲区中
#             RL.store_transition(obervation, action, reward, obervation_)  # store_transition是一个函数，用于存储经验
#
#             # 当与环境交互的步数超过200步时，开始学习DQN网络参数，每5步学习一次
#             if (step > 200) and (step % 5 == 0):  # 当满足条件时（步数大于200并且步数能被5整除），执行learn函数进行学习
#                 RL.learn()  # learn是一个函数，用于更新DQN网络的参数
#
#             # 更新当前观测为下一个观测，为下一步的交互做准备
#             obervation = obervation_
#
#             # 如果任务结束，跳出循环，进行下一个episode的任务
#             if done:
#                 break
#             step += 1  # 步数加一
#     print("end of game")  # 打印任务结束的信息
#     env.destroy()  # 销毁环境，可能用于释放资源或关闭窗口等操作
#
#
# if __name__ == '__main__':  # 如果这个文件是直接运行的（而不是被导入的），那么执行以下代码
#     env = Maze()  # 创建一个Maze环境的实例，并赋值给env变量
#     RL = DeepQNetwork(  # 创建一个DeepQNetwork的实例，并赋值给RL变量
#         env.n_actions, env.n_features,  # 输入env的行动数（n_actions）和特征数（n_features）作为网络的输入维度
#         learning_rate=0.01,  # 设置学习率为0.01
#         reward_decay=0.9,  # 设置奖励衰减因子为0.9（可能用于奖励的折扣）
#         e_greedy=0.9,  # 设置epsilon-greedy策略的epsilon值为0.9，即有90%的概率选择随机动作
#         replace_target_iter=200,  # 设置目标网络替换频率为200步
#         memory_size=2000,  # 设置经验回放缓冲区的大小为2000条经验
#         # output_graph=True  # 可能用于可视化网络结构，但在此代码中未使用到此选项
#     )
# env.after(100, update)  # 在env运行100步后，执行update函数进行更新（可能用于更新环境或DQN网络的参数等）
# env.mainloop()  # 开始运行环境的主循环，与DQN网络进行交互完成任务
# RL.plot_cost()  # 在完成所有任务后，绘制DQN网络训练过程中成本的变化曲线
#
#
#
#
#
#
#
#
