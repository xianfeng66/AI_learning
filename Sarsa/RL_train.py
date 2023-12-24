#代码实战Q-Learning智能体选择行为

import numpy as np
import pandas as pd


class RL:         #QLearningAgent
    def __init__(self, actions, learning_rate = 0.01, reward_decay = 0.9, e_greedy = 0.9):
        self.actions = actions # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self,observation):
        self.check_state_exist(observation)
        #choose best action
        if np.random.uniform() < self.epsilon:
            #choose best action
            state_action = self.q_table.loc[observation,:]
            #some actions may have the same value, randomly choose one in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action
class SarsaTable(RL):
    def __init__(self,actions,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9):
        super(SarsaTable,self).__init__(actions,learning_rate,reward_decay,e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exists(s_)
        q_predict = self.q_table.loc[s,a]
        if s_!= 'terminal':
            q_target = r +self.gamma * self.q_table.loc[s, :].max()  #next state is not terminal
        else:
            q_target = r # next state is terminal
        self.q_table.loc[s,a] += self.lr*(q_target - q_predict)  #update


class SarsaLambdaTable(RL)：
     def __init__(self,actions,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9,trace_decay=0.9):
         super(SarsaTable,self).__init__(actions,learning_rate,reward_decay,e_greedy)

         self.lambda_ = trace_decay
         #在对应观测状态的state对应的action的位置加1
         self.eligibility_trace = self.q_table.copy()


    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            to_be_append = pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            self.q_table = self.q_table.append(to_be_append)

            self.eligibility_trace = self.eligibility_trace.append(to_be_append)


    def learn(self, s, a, r, s_,a_):
        self.check_state_exists(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma +*self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        error = q_target - q_predict

        #Method1
        # self.eligibility_trace.loc[s,a] += 1

        #Method2
        self.eligibility_trace.loc[s,:] *= 0
        self.eligibility_trace.loc[s,a] = 1

        #跟新Q-tables,更新v
        self.q_table += self.lr * error * self.eligibility_trace

        #衰减
        self.eligibility_trace*=self.gamma*self.lambda_


# # 导入numpy库，为后续的数值计算提供支持
# import numpy as np
# # 导入pandas库，用于数据分析和处理，后面的DataFrame就是它的一个重要应用
# import pandas as pd
#
#
# # 定义一个名为QLearningAgent的类，该类用于实现基于Q学习的智能体行为选择和更新
# class QLearningAgent:
#     # 初始化函数，接收四个参数：actions（所有可能的行为）、learning_rate（学习率）、reward_decay（奖励衰减因子）和e_greedy（贪婪度）
#     def __init__(self, actions, learning_rate=0.01, reward_deecay=0.9, e_greedy=0.9):
#         # 将actions保存为类的属性，它是一个行为列表
#         self.actions = actions  # a list
#         # 保存学习率
#         self.lr = learning_rate
#         # 保存奖励衰减因子
#         self.gamma = reward_deecay
#         # 保存贪婪度
#         self.epsilon = e_greedy
#         # 创建一个数据框作为Q表，列是所有的行为，数据类型是float64，初始值都为0
#         self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
#
#         # 检查状态是否存在，如果不在则添加新的状态到Q表中
#
#     def check_state_exist(self, state):
#         if state not in self.q_table.index:
#             # 如果状态不存在，则在新状态后面追加一行，所有的行为值都设为0
#             # append new state to q table
#             self.q_table = self.q_table.append(
#                 pd.Series(
#                     [0] * len(self.actions),  # 新增的行中每个行为值都设为0
#                     index=self.q_table.columns,  # 行索引设为原来的列名（即行为）
#                     name=state,  # 新增行的名字设为当前的状态名
#                 )
#             )
#
#             # 基于当前的状态选择一个行为，先检查状态是否存在，然后根据epsilon随机选择行为（epsilon greedy策略）
#
#     def choose_action(self, observation):
#         self.check_state_exist(observation)  # 检查状态是否存在，不存在则添加到Q表中
#         # 根据epsilon随机选择一个行为，如果小于epsilon，则随机选择一个行为，否则选择最好的行为
#         if np.random.uniform() < self.epsilon:  # choose best action
#             # 选择最好的行为，选择所有等于最大值的索引，然后随机选择一个作为行为
#             state_action = self.q_table.loc[observation, :]  # 获取当前状态的所有行为值
#             action = np.random.choice(state_action[state_action == np.max(state_action)].index)  # 选择最大的行为值对应的索引作为行为
#         else:  # choose random action
#             # 如果大于epsilon，则随机选择一个行为
#             action = np.random.choice(self.actions)  # 在所有行为中随机选择一个作为行为
#         return action  # 返回选择的行为
#
#
#     # 定义SarsaTable类，这个类继承自RL类
#     class SarsaTable(RL):
#         # 定义类的初始化函数
#         def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
#             # 调用父类的初始化函数，传递参数
#             super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
#
#             # 定义学习函数
#
#         def learn(self, s, a, r, s_):
#             # 调用父类的方法，检查下一个状态是否存在，如果不存在，则进行添加
#             self.check_state_exists(s_)
#             # 获取当前状态行为对应的Q值
#             q_predict = self.q_table.loc[s, a]
#             if s_ != 'terminal':  # 如果下一个状态不是终止状态
#                 # 计算目标Q值，下一个状态的最大Q值是由奖励加上折扣因子乘以下一个状态的Q值得到的
#                 q_target = r + self.gamma * self.q_table.loc[s, :].max()
#             else:  # 如果下一个状态是终止状态
#                 # 目标Q值就是奖励
#                 q_target = r
#                 # 更新Q值，学习率乘以目标Q值减去预测Q值
#             self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
#
#     # 定义SarsaLambdaTable类，这个类继承自RL类
#     class SarsaLambdaTable(RL):
#         # 定义类的初始化函数
#         def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
#             # 调用父类的初始化函数，传递参数
#             super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
#             # 初始化trace_decay参数，用于更新eligibility trace
#             self.lambda_ = trace_decay
#             # 初始化eligibility trace，与q_table具有相同的索引和列，初始值都为0
#             self.eligibility_trace = self.q_table.copy()
#
#             # 定义check_state_exists函数，这个函数检查给定的状态是否存在于q_table中，如果不存在则进行添加
#
#         def check_state_exist(self, state):
#             if state not in self.q_table.index:  # 如果状态不在q_table的索引中
#                 # 创建一个新的Series，其中的元素都是0，与q_table具有相同的列，但是新的状态名作为名称
#                 to_be_append = pd.Series(
#                     [0] * len(self.actions),
#                     index=self.q_table.columns,
#                     name=state,
#                 )
#                 # 将新的Series添加到q_table中，也将其添加到eligibility trace中
#                 self.q_table = self.q_table.append(to_be_append)
#                 self.eligibility_trace = self.eligibility_trace.append(to_be_append)
#
#                 # 定义学习函数，这个函数根据Sarsa(λ)算法进行更新
#
#         def learn(self, s, a, r, s_, a_):
#             # 调用父类的方法，检查下一个状态是否存在，如果不存在，则进行添加
#             self.check_state_exists(s_)
#             # 获取当前状态行为对应的Q值
#             q_predict = self.q_table.loc[s, a]
#             if s_ != 'terminal':  # 如果下一个状态不是终止状态
#                 # 计算目标Q值，下一个状态的最大Q值是由奖励加上折扣因子乘以下一个状态的Q值得到的
#                 q_target = r + self.gamma * self.q_table.loc[s_, a_]
#             else:  # 如果下一个状态是终止状态
#                 # 目标Q值就是奖励
#                 q_target = r
#                 # 计算误差，目标Q值减去预测Q值
#             error = q_target - q_predict
#
#             # Method 2 是该段代码的标题或者说明，表示这是Sarsa(λ)算法的第二种实现方式
#
#             # 初始化eligibility_trace 为q_table的副本，所有值都设为0
#             # 该变量用于存储每个状态-动作对的“资格”，即用于衡量在更新Q值时，每个状态-动作对的“重要性”
#             self.eligibility_trace = self.q_table.copy()
#
#             # 定义一个检查函数，检查给定的状态是否存在于q_table中，如果不存在则进行添加
#             # 如果状态不在q_table的索引中，那么就创建一个新的Series，其中的元素都是0，与q_table具有相同的列，但是新的状态名作为名称，然后将其添加到q_table和eligibility_trace中
#             self.check_state_exist(state)
#
#             # 在eligibility_trace中将给定状态对应的所有动作的资格都设为0，即将它们重新设为“非资格”状态
#             # 只有给定状态采取特定动作时，对应的资格才会被设置为1，表示该动作具有资格进行学习
#             self.eligibility_trace.loc[s, :] *= 0
#             self.eligibility_trace.loc[s, a] = 1
#
#             # 根据误差更新q_table，同时乘以eligibility_trace，使得具有资格的动作将获得更大的学习权重
#             # 同时对eligibility_trace进行衰减，表示随着时间的推移，旧的、已经不再更新的状态-动作对的资格将逐渐减小
#             # lr是学习率，error是目标值与预测值的误差，gamma是折扣因子，lambda_是trace的衰减因子
#             self.q_table += self.lr * error * self.eligibility_trace
#             self.eligibility_trace *= self.gamma * self.lambda_



# 当然可以。这段代码定义了一个名为QLearningAgent的类，它实现了一种被称为Q学习的强化学习算法的一部分。让我们逐行解释这段代码：
#
# import numpy as np - 导入numpy库，这是一个用于大规模数值计算的库。
#
# import pandas as pd - 导入pandas库，这是一个用于数据操作和分析的库。
#
# class QLearningAgent: - 定义了一个名为QLearningAgent的类。
#
# def __init__(self, actions, learning_rate = 0.01, reward_deecay = 0.9, e_greedy = 0.9): - 这是类的初始化函数，它接收四个参数：
#
# actions：一个包含所有可能行为的列表。
# learning_rate：学习率，用于更新Q值。默认值为0.01。
# reward_decay：奖励衰减因子，可能是一个在0到1之间的值，用于在每次更新Q值时逐渐减少奖励。默认值为0.9。
# e_greedy：贪婪度，一个在0到1之间的值。当它接近1时，智能体更倾向于选择具有最大Q值的动作，当它接近0时，智能体更随机。默认值为0.9。
# self.actions = actions - 将actions保存为类的属性，表示所有可能的行为。
#
# self.lr = learning_rate - 保存学习率的属性。
#
# self.gamma = reward_deecay - 保存奖励衰减因子的属性。
#
# self.epsilon = e_greedy - 保存贪婪度的属性。
#
# self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) - 创建一个pandas DataFrame作为Q表。它的列是所有的行为，数据类型是float64，初始值都为0。
#
# def check_state_exist(self, state): - 定义一个函数来检查给定的状态是否存在于Q表中。
#
# if state not in self.q_table.index: - 检查状态是否在Q表的索引中。
#
# self.q_table = self.q_table.append(...) - 如果状态不存在，则在新状态后面追加一行，所有的行为值都设为0。
#
# def choose_action(self, observation): - 定义一个函数来基于给定的观察选择一个行为。
#
# self.check_state_exist(observation) - 检查给定的观察是否存在于Q表中。
#
# if np.random.uniform() < self.epsilon: - 如果随机数小于贪婪度，则选择具有最大Q值的动作（epsilon greedy策略）。
#
# state_action = self.q_table.loc[observation,:] - 从Q表中获取当前状态的所有行为值。
#
# action = np.random.choice(state_action[state_action == np.max(state_action)].index) - 选择具有最大Q值的动作的索引作为行为。
#
# else: - 如果随机数大于贪婪度，则随机选择一个行为。
#
# action = np.random.choice(self.actions) - 在所有行为中随机选择一个作为行为。
#
# return action - 返回选择的行为。
#
# 这个类的主要目的是实现Q学习算法的一个关键部分，即基于Q表选择行为。它维护一个Q表来存储每个状态和行为的Q值，并使用epsilon greedy策略在选择行为时平衡探索和利用。





# 这段代码是使用Python编写的，它定义了两个类，SarsaTable和SarsaLambdaTable，这两个类
# 都是强化学习算法的实现，用于通过Q学习（Q-learning）来让一个智能体学习如何在环境中采取行动
# 以获得最大的累积奖励。
#
# SarsaTable类：
#
# __init__函数：初始化一个名为SarsaTable的对象，并接受四个参数：
# actions（所有可能的行为的列表），learning_rate（学习率，用于更新Q值），
# reward_decay（奖励衰减因子，可能是一个在0到1之间的值，用于在每次更新Q值时逐渐减少奖励），
# 和e_greedy（贪婪度，一个在0到1之间的值）。它也调用其父类RL的初始化函数来进行进一步的初始化。
# learn函数：根据当前状态和行为，以及下一个状态和下一个行为（如果有的话），更新Q表中的Q值。
# 首先检查下一个状态是否存在，如果不存在，则将其添加到Q表中。然后计算预测的Q值和目标Q值，
# 并使用这些值来更新Q表中的Q值。
# SarsaLambdaTable类：
#
# __init__函数：初始化一个名为SarsaLambdaTable的对象，并接受五个参数：
# actions（所有可能的行为的列表），learning_rate（学习率，用于更新Q值），
# reward_decay（奖励衰减因子，可能是一个在0到1之间的值，用于在每次更新Q值时逐渐减少奖励），
# e_greedy（贪婪度，一个在0到1之间的值），和trace_decay（追踪衰减因子）。
# 它也调用其父类RL的初始化函数来进行进一步的初始化。
# 它还创建了一个名为eligibility_trace的属性，
# 这是一个与Q表具有相同结构的表，用于存储追踪信息。
# check_state_exist函数：这个函数检查给定的状态是否存在于Q表中。
# 如果不存在，则将新状态添加到Q表和eligibility_trace中。
# learn函数：与SarsaTable中的learn函数类似，但有一个额外的参数a_，
# 代表下一个状态的行动。它还使用了eligibility_trace来更新Q值。在更新Q值之前，
# 它将eligibility_trace矩阵中对应于当前状态的所有行动的元素设置为0，因为在更新Q值时，
# 我们只关心当前状态的特定行动。
# 请注意，代码中似乎存在一些错误和不完整的部分。例如，在SarsaLambdaTable类的learn函数中，
# 有一行代码 self.q_table.loc[s_,a_] 是不完整的，它缺少对 q_table 的索引。
# 另外，这段代码中没有定义 RL 类及其 __init__ 函数，也没有定义 q_table 和 gamma 这些属性是在哪里定义的。这些都需要在完整的代码中加以解决。