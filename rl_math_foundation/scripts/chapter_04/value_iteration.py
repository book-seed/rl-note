import sys
import time
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import grid_env


class class_value_iteration:
    def __init__(self, env: grid_env.GridEnv):
        self.gama = 0.9
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size**2
        self.reward_space_size, self.reward_list = (
            len(self.env.reward_list),
            self.env.reward_list,
        )
        self.state_value = np.zeros(shape=self.state_space_size)
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))

        self.mean_policy = (
            np.ones(shape=(self.state_space_size, self.action_space_size))
            / self.action_space_size
        )
        self.policy = self.mean_policy.copy()
        self.writer = SummaryWriter("../logs")

        print(
            f"action_space_size: {self.action_space_size} state_space_size: {self.state_space_size}"
        )
        print(
            f"state_value.shape:{self.state_value.shape} , qvalue.shape:{self.qvalue.shape} , mean_policy.shape:{self.mean_policy.shape}"
        )
        print(
            f"self.reward_space_size:{self.reward_space_size}, self.reward_list:{self.reward_list}, is non-forbidden area, target area, forbidden area, overflow respectively."
        )
        print("----------------------------------------------------------------")

    def value_iteration(self, tolerance=0.001, steps=100):
        """
        迭代求解最优贝尔曼公式 得到 最优state value tolerance 和 steps 满足其一即可
        :param tolerance: 当 前后 state_value 的范数小于tolerance 则认为state_value 已经收敛
        :param steps: 当迭代次数大于step时 停止 建议将此变量设置大一些
        :return: 剩余迭代次数
        """
        # 初始化 V0 为 1
        state_value_k = np.ones(self.state_space_size)
        while (
            # ‖d‖₁ = |v0'-v0| + |v1'-v1| + ... + |v8'-v8|
            np.linalg.norm(state_value_k - self.state_value, ord=1) > tolerance
            and steps > 0
        ):
            steps -= 1
            self.state_value = state_value_k.copy()
            """
            是普通 policy_improvement 的变种 相当于是值迭代算法 也可以 供策略迭代使用 做策略迭代时不需要 接收第二个返回值
            更新 qvalue, qvalue[state,action]=reward+value[next_state]
            找到 state 处的 action*：action* = arg max(qvalue[state,action]) 即最优action即最大qvalue对应的action
            更新 policy ：将 action*的概率设为1 其他action的概率设为0 这是一个greedy policy
            :param: state_value: policy对应的state value
            :return: improved policy, 以及迭代下一步的state_value
            """
            # 方法初始化了一个新的策略 policy，所有状态的所有动作的概率都被设置为0
            policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
            q_table = np.zeros(shape=(self.state_space_size, self.action_space_size))
            for state in range(self.state_space_size):
                qvalue_list = []
                for action in range(self.action_space_size):
                    qvalue = 0
                    for i in range(self.reward_space_size):
                        qvalue += self.reward_list[i] * self.env.Rsa[state, action, i]

                    for next_state in range(self.state_space_size):
                        qvalue += (
                            self.gama
                            * self.env.Psa[state, action, next_state]
                            * state_value_k[next_state]
                        )
                    qvalue_list.append(qvalue)
                # print("qvalue_list:",qvalue_list)
                q_table[state, :] = qvalue_list.copy()

                state_value_k[state] = max(qvalue_list)  # 取该state 的最大state value
                action_star = qvalue_list.index(
                    max(qvalue_list)
                )  # 取该state 的最大state value对应的action
                policy[state, action_star] = 1  # 更新策略，贪婪算法
            print(f"q_table:{q_table}")
            self.policy = policy
        return steps


if __name__ == "__main__":
    print("-----Begin!-----")
    gird_world2x2 = grid_env.GridEnv(
        size=3, target=[2, 2], forbidden=[[1, 0], [2, 1]], render_mode=""
    )
    solver = class_value_iteration(gird_world2x2)
    start_time = time.time()

    # run value iteration algorithm for a fixed number of steps
    demand_step = 1000
    remaining_step = solver.value_iteration(tolerance=0.1, steps=demand_step)
