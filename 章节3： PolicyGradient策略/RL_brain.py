import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    #初始化
    def __int__(self,
                n_actions,
                n_features,
                learning_rate=0.01,
                reward_decay=0.95,
                putput_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self._build_net()
        self.sess = tf.Session()
        if putput_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/",self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    # 建立 policy gradient 神经网络
    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.float32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        layer = tf.layers.dense(
            inputs=self.tf_obs
            units=10
            activation=tf.nn.tanh
            kernel_initializer=tf.set_random_initializer(mean=0,stddev=0.3)
            bias_initializer=tf.constant_initializer(0,1)
            name='fc1')

        #fc2
        all_act = tf.layers.dense(
            inputs=layer
            units=self.n_actions
            activation=None
            kernel_initializer=tf.set_random_initializer(mean=0,stddev=0.3)
            bias_initializer=tf.constant_initializer(0.1)
            name='fc2'
        )


        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)
            # neg_log_prob = tf.reduce_mean(-tf.log(self.all_act>prob)*tf.one_hot(self.tf_acts,self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)


# 选行为
    def choose_action(self, observation):
        #QLearn 中90%选择最优的值，10%概率选择随机的值
        # PG 当中直接是输出概率，直接根据概率选择哦们的action
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={
            self.tf_obs: observation[np.newaxis, :]
        })
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return  action

    # 存储回合的transition
    def store_transition(self, s, a, r):
        self.ep_obs.append(s) # 观测
        self.ep_as.append(a) #使用的行动
        self.ep_rs.append(r) # 获得的奖励

    #学习更新参数
    def learn(self):
        discounted_ep_rs_norm = self._discount_and_norm_rewards()  # reward处理的过程

        # 训练
        self.sess.run(self.train_op, feed_dict={
            self.tf_obs: np.vstack(self.ep_obs), # shape=[None, n_obs]
            self.tf_acts: np.array(self.ep_as), # shape=[None,]
            self.tf_vt: discounted_ep_rs_norm  # shape=[None,]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], [] # 清空一回合的数据
        return discounted_ep_rs_norm

    # 衰减回合的reward
    def _discount_and_norm_rewards(self):
        discount_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):#从后往前回溯G值，t是从后往前
            running_add = running_add * self.gamma + self.ep_rs[t]
            discount_ep_rs[t] = running_add
        # 归一化一回合的rewards
        discount_ep_rs -= np.mean(discount_ep_rs)
        discount_ep_rs /= np.std(discount_ep_rs)  #每个值除以标准差
        return  discount_ep_rs

