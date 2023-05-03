import pandas as pd
import numpy as np
from scipy.optimize import minimize


# 定义CVaR目标函数
def cvar_objective(weights, returns, alpha):
    portfolio_return = np.dot(weights, returns.mean())
    portfolio_var = np.dot(weights.T, np.dot(returns.cov(), weights))
    portfolio_cvar = -portfolio_return + (1/alpha) * np.mean(
        [-portfolio_return + r for r in returns @ weights if r < portfolio_return])
    return portfolio_cvar


# 读取数据
data = pd.read_excel("C:/Users/86183/Desktop/副本sheet.xlsx", index_col=0, header=0)
returns = data.pct_change().dropna()

# 设定初始权重和边界
num_assets = returns.shape[1]
initial_guess = np.array([1/num_assets]*num_assets)
bounds = [(0, 1) for i in range(num_assets)]
constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

# 计算有效边界
frontier_y = np.linspace(returns.min().min(), returns.max().max(), 100)
frontier_x = []
for possible_return in frontier_y:
    constraints.append({'type': 'eq', 'fun': lambda x, y=possible_return: np.dot(x, returns.mean()) - y})
    result = minimize(cvar_objective, initial_guess, args=(returns, 0.95), method='SLSQP', bounds=bounds, constraints=constraints)
    frontier_x.append(result['fun'])
    constraints.pop()

# 打印有效边界
import matplotlib.pyplot as plt
plt.plot(frontier_x, frontier_y)
plt.title('Mean-CVaR Portfolio Optimization')
plt.xlabel('Expected CVaR')
plt.ylabel('Expected Return')
plt.show()
