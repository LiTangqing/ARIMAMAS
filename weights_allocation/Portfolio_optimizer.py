import numpy as np
import pandas as pd
from scipy.optimize import minimize

def sharpe_ratio(daily_returns):
    # get volatility
    vola_daily = np.std(daily_returns)
    if vola_daily <= 0.000001: # prevent div by 0
        vola_daily += 0.000001
    # get returns
    total_returns = np.product(1 + daily_returns)
    returns_daily = np.exp(np.log(total_returns)/daily_returns.shape[0]) - 1
    return returns_daily / vola_daily
    
def sortino_ratio(daily_returns):
    # get volatility
    downsideReturns = daily_returns.copy()
    downsideReturns[downsideReturns > 0] = 0
    downsideVola = np.std(downsideReturns)
    if downsideVola <= 0.000001: # prevent div by 0
        downsideVola += 0.000001
    # get returns
    total_returns = np.product(1 + daily_returns)
    returns_daily = np.exp(np.log(total_returns)/daily_returns.shape[0]) - 1
    return returns_daily / downsideVola

def slippage_costs(high, low, close, delta_weights, factor=0.05):
    """Taken as (High - Low) * factor, measured in absolute costs."""
    slippage = (high - low) / close * 0.05
    return abs(delta_weights * slippage)

def new_portfolio_equity(portfolio_returns, weights, old_portfolio_equity):
    """Given ytd's portfolio equity, weights and today's returns, find today's total equity.
    
    Assumption: weights[0] is weight of CASH asset.
    """
    future_equity = max(np.sum(portfolio_returns) * old_portfolio_equity * (1 - weights[0]), 0)
    return pd.Series(future_equity + weights[0] * old_portfolio_equity)

def single_sim_sharpe(portfolio, new_weights, old_weights, portfolio_val, pred_prices, index):
    """Computes the estimated sharpe ratio for tomorrow given weights and predictions."""
    data = portfolio
     
    # index of days for convenience of accessing data's values 
    TMR, TODAY, YTD = index+1, index, index-1
    
    # today's sharpe ratio (compute using past 5 days returns/std dev)
    # array of each stock's pct return today
    today_ret = (data.loc[TODAY, 'CLOSE'] - data.loc[YTD, 'CLOSE']) / data.loc[YTD, 'CLOSE']
    # array of weighted stock pct returns
    today_portfolio_ret = old_weights[1:] * today_ret # Drop CASH (first weight)
    # add today's new portfolio value to Series of portfolio values
    today_portfolio_val = portfolio_val.append(
        new_portfolio_equity(portfolio_returns=today_portfolio_ret,
                             weights=old_weights,
                             old_portfolio_equity=portfolio_val.iloc[-1]),
        ignore_index=True
    )
    # sharpe of pct returns over past 5 days (inc. today)
    today_sharpe = sharpe_ratio(today_portfolio_val.pct_change().tail(5))
    
    delta_weights = new_weights[1:] - old_weights[1:] # Cash has no slippage
    # Use today's high-low to estimate slippage tomorrow
    slippage = slippage_costs(data.loc[TODAY, 'HIGH'], 
                              data.loc[TODAY, 'LOW'],
                              data.loc[YTD, 'CLOSE'],
                              delta_weights)
    
    # find tomorrow's returns based on new weights and predicted prices
    # ASSUMPTION: tomorrow's OPEN = today's CLOSE
    # array of each stock's predicted pct return tomorrow
    pred_ret = (pred_prices.loc[TMR, :] - data.loc[TODAY, 'CLOSE']) / data.loc[TODAY, 'CLOSE'] # % change
    # array of predicted weighted stock pct returns, deducting slippage for each stock
    pred_portfolio_ret = new_weights[1:] * pred_ret - slippage
    # sharpe of past 4 days (inc. today) + tomorrows predicted portfolio return
    pred_portfolio_val = today_portfolio_val.append(
        new_portfolio_equity(portfolio_returns=pred_portfolio_ret,
                             weights=new_weights,
                             old_portfolio_equity=today_portfolio_val.iloc[-1]),
        ignore_index=True
    )
    pred_sharpe = sharpe_ratio(pred_portfolio_val.pct_change().tail(5))
    
    # maximize the difference here to greedily optimize
    return pred_sharpe  

def neg_obj_function(new_weights, data, old_weights, portfolio_val, pred_prices, index):
    return -single_sim_sharpe(data, new_weights, old_weights,
                              portfolio_val, pred_prices, index)

def max_diff_sharpe_ratio(data, old_weights, portfolio_val, pred_prices, index):
    num_assets = old_weights.shape[0]
    args = (data, old_weights, portfolio_val, pred_prices, index)
    
    # constraint: sum of weight = 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # for each future, search from -1 to 1
    bound = (-1.0, 1.0) 
    # Cash is first weight, bound within 0,1
    bounds = ((0.0, 1.0),) + tuple(bound for asset in range(1, num_assets))
    # to aid convergence if necessary, 2-3mins for 200iters
    opts = {'maxiter':200}
    # run optimisation:
    # maximize the sharpe = minimize the neg value of it
    result = minimize(neg_obj_function, np.array(num_assets*[1./num_assets,]), 
                      args=args, method='SLSQP', options=opts,
                      bounds=bounds, constraints=constraints)
    return result 