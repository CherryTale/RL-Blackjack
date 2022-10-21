from random import shuffle, random, randint
import numpy as np
from matplotlib import pyplot as plt

cards = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7,
         7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
winlose = []


def mySum(currentCards):
    n = sum(currentCards)
    if n < 12 and 1 in currentCards:
        n += 10
    return n


def train(episodes, alpha, gamma, epsilon):  # 定义生成事件数量

    # q_sa => [usableAcePlayer, playerSum-12, dealerCard_top-1, hit?1:0]
    q_sa = np.random.rand(2, 10, 10, 2)

    # loop and train
    while episodes > 0:
        shuffle(cards)
        playerCards = [cards[0], cards[1]]
        dealerCards = [cards[2], cards[3]]
        # 记录取牌位置
        p_cards = 4

        # 记录usableAce
        usableAce = 0
        if playerCards[0] == 1 or playerCards[1] == 1:
            usableAce = 1

        # 记录庄家顶部牌
        dealerCard_top = dealerCards[0]

        # 如果玩家手中牌和小于12, 直接取牌
        while mySum(playerCards) < 12:
            playerCards.append(cards[p_cards])
            p_cards += 1
            if playerCards[-1] == 1 and sum(playerCards) < 12:
                usableAce = 1

        if mySum(playerCards) == 21 or mySum(dealerCards) == 21:
            continue

        # 状态序列
        s = [usableAce, mySum(playerCards)-12, dealerCard_top-1]
        action = q_sa[s[0], s[1], s[2], 0] <= q_sa[s[0], s[1], s[2], 1]

        while True:
            # 玩家根据贪婪策略值是否取牌, 取q值最大执行动作
            if (action):

                playerCards.append(cards[p_cards])
                p_cards += 1

                if mySum(playerCards) > 21:
                    q_sa[s[0], s[1], s[2], 1] += alpha * \
                        (-1-q_sa[s[0], s[1], s[2], 1])
                    winlose.append(0)
                    break
                else:
                    s_prime = [usableAce, mySum(
                        playerCards)-12, dealerCard_top-1]
                    action = 1 if q_sa[s_prime[0], s_prime[1], s_prime[2],
                                       0] <= q_sa[s_prime[0], s_prime[1], s_prime[2], 1] else 0
                    q_sa[s[0], s[1], s[2], 1] += alpha*(
                        gamma*q_sa[s_prime[0], s_prime[1], s_prime[2], action]-q_sa[s[0], s[1], s[2], 1])
                    s = s_prime

            else:
                # dealer 取牌
                while mySum(dealerCards) < 17:
                    dealerCards.append(cards[p_cards])
                    p_cards += 1

                # 判断胜负 给出reward
                if mySum(dealerCards) > 21:
                    q_sa[s[0], s[1], s[2], action] += alpha * \
                        (1-q_sa[s[0], s[1], s[2], 0])
                    winlose.append(1)
                else:
                    if mySum(playerCards) > mySum(dealerCards):
                        q_sa[s[0], s[1], s[2], 0] += alpha * \
                            (1-q_sa[s[0], s[1], s[2], 0])
                        winlose.append(1)
                    if mySum(playerCards) == mySum(dealerCards):
                        q_sa[s[0], s[1], s[2], 0] += alpha * \
                            (-q_sa[s[0], s[1], s[2], 0])
                        winlose.append(0)
                    if mySum(playerCards) < mySum(dealerCards):
                        q_sa[s[0], s[1], s[2], 0] += alpha * \
                            (-1-q_sa[s[0], s[1], s[2], 0])
                        winlose.append(0)

                break

        episodes -= 1

    return q_sa


q_sa = train(500000, 0.01, 0.5, 0)

# 画图
plt.figure("Result")

fig = plt.subplot(2, 2, 1)
fig.set_title("No usable ace")
fig.set_xticks(np.arange(10), np.arange(1, 11))
fig.set_yticks(np.arange(10), np.arange(12, 22))
im=fig.imshow(q_sa[0, :, :, 1] - q_sa[0, :, :, 0],
           cmap="coolwarm", norm=plt.Normalize(vmin=-1, vmax=1))
plt.colorbar(im)

fig = plt.subplot(2, 2, 2)
fig.set_title("Usable ace")
fig.set_xticks(np.arange(10), np.arange(1, 11))
fig.set_yticks(np.arange(10), np.arange(12, 22))
im=fig.imshow(q_sa[1, :, :, 1] - q_sa[1, :, :, 0],
           cmap="coolwarm", norm=plt.Normalize(vmin=-1, vmax=1))
plt.colorbar(im)

v_s = np.zeros((2, 10, 10))
for i in range(2):
    for j in range(10):
        for k in range(10):
            v_s[i, j, k] = max(q_sa[i, j, k, :])

fig3d = plt.subplot(2, 2, 3, projection='3d')
x = [i for i in range(10)]
y = [i for i in range(10)]
x, y = np.meshgrid(x, y)
z = v_s[0]
fig3d.set_xticks(np.arange(10), np.arange(1, 11))
fig3d.set_yticks(np.arange(10), np.arange(12, 22))
fig3d.set_zlim3d(zmin=-1, zmax=1)
fig3d.plot_surface(x, y, z, cmap="viridis")

fig3d = plt.subplot(2, 2, 4, projection='3d')
x = [i for i in range(10)]
y = [i for i in range(10)]
x, y = np.meshgrid(x, y)
z = v_s[1]
fig3d.set_xticks(np.arange(10), np.arange(1, 11))
fig3d.set_yticks(np.arange(10), np.arange(12, 22))
fig3d.set_zlim3d(zmin=-1, zmax=1)
fig3d.plot_surface(x, y, z, cmap="viridis")

plt.figure("Winrate")

x = []
y = []
for i in range(len(winlose)-10000):
    x.append(i)
    y.append(sum(winlose[i:i+10000]))
plt.plot(x, y)

plt.show()
