"""Helper functions for optimizing trading strategy."""

# Imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize

__version__ = '1.0'

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
    profit_value = np.sum(portfolio_returns) * old_portfolio_equity * (1 - weights[0])
    return pd.Series(max(profit_value + old_portfolio_equity, 0.01))

def calc_portfolio_returns(new_prices, old_prices, weights):
    """Compute weighted arithmetic return of each asset in new_prices."""
    ret = (new_prices - old_prices)/old_prices
    return weights * ret

def single_sim_sharpe(portfolio, new_weights, old_weights, portfolio_val, pred_prices, index):
    """copy from the above, except return the sharpe instead of the difference"""
    """Computes the estimated sharpe ratio for tomorrow given weights and predictions."""
    data = portfolio
     
    # index of today for convenience of accessing data's values 
    TMR, TODAY, YTD = index+1, index, index-1
    
#     # today's sharpe ratio (compute using past 5 days returns/std dev)
#     # array of weighted stock pct returns
#     today_portfolio_ret = calc_portfolio_returns(data.loc[TODAY, 'CLOSE'],
#                                                  data.loc[YTD, 'CLOSE'],
#                                                  old_weights[1:]) # Drop CASH (first weight)
#     # add today's new portfolio value to Series of portfolio values
#     today_portfolio_val = portfolio_val.append(
#         new_portfolio_equity(portfolio_returns=today_portfolio_ret,
#                              weights=old_weights,
#                              old_portfolio_equity=portfolio_val.iloc[-1]),
#         ignore_index=True
#     )
#     # sharpe of pct returns over past 5 days (inc. today)
#     today_sharpe = sharpe_ratio(today_portfolio_val.pct_change().tail(5))
    
    delta_weights = new_weights[1:] - old_weights[1:] # Cash has no slippage
    # Use today's high-low to estimate slippage tomorrow
    slippage = slippage_costs(data['HIGH'][TODAY],#.loc[TODAY, 'HIGH'], 
                              data['LOW'][TODAY],#.loc[TODAY, 'LOW'],
                              data['CLOSE'][YTD],#.loc[YTD, 'CLOSE'],
                              delta_weights)
    
    # find tomorrow's returns based on new weights and predicted prices
    # CRITICAL ASSUMPTION: tomorrow's OPEN = today's CLOSE
    # array of predicted weighted stock pct returns, deducting slippage for each stock
    pred_portfolio_ret = calc_portfolio_returns(pred_prices,#.loc[TMR, :],
                                                data['CLOSE'][TODAY],#.loc[TODAY, 'CLOSE'],
                                                new_weights[1:])
    pred_portfolio_ret -= slippage
    # sharpe of past 4 days (inc. today) + tomorrows predicted portfolio return
    pred_portfolio_val = portfolio_val.append(
        new_portfolio_equity(portfolio_returns=pred_portfolio_ret,
                             weights=new_weights,
                             old_portfolio_equity=portfolio_val.iloc[-1]),
        ignore_index=True
    )
    pred_sharpe = sharpe_ratio(pred_portfolio_val.pct_change().tail(5))
    
    # maximize the difference here to greedily optimize
    return pred_sharpe.iloc[-1] if pred_sharpe.shape[0] > 1 else pred_sharpe

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
