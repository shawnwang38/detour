"""
TWAP Cost Estimation Algorithm
"""

import json
import requests
import numpy as np
from typing import Dict, Tuple, Optional
import sys
import os
import math
import copy


def get_market_data(symbol: str, use_live: bool = True) -> Dict:
    """
    Fetch order book and market statistics.
    
    Returns:
        Dict containing order_book, daily_volume, mid_price, spread
    """
    # Get order book
    if use_live:
        url = f"https://api.gemini.com/v1/book/{symbol}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            book_data = response.json()
            
            for side in ['bids', 'asks']:
                book_data[side] = [{
                    'price': float(order['price']),
                    'amount': float(order['amount'])
                } for order in book_data[side]]
        except Exception as e:
            print(f"Error fetching order book: {e}")
            return None
    else:
        # Use cached data
        os.makedirs('test_data', exist_ok=True)
        filename = f'test_data/{symbol}_snapshot.json'
        
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                book_data = json.load(f)
        else:
            # Fetch once and save
            book_data = get_market_data(symbol, use_live=True)['order_book']
            with open(filename, 'w') as f:
                json.dump(book_data, f, indent=2)
    
    # Get ticker data for volume
    ticker_url = f"https://api.gemini.com/v1/pubticker/{symbol}"
    try:
        response = requests.get(ticker_url)
        response.raise_for_status()
        ticker = response.json()
        
        # Extract daily volume
        volume_keys = list(ticker['volume'].keys())
        daily_volume = float(ticker['volume'][volume_keys[0]]) if volume_keys else 0
    except:
        daily_volume = 0
    
    # Calculate metrics
    if book_data['bids'] and book_data['asks']:
        best_bid = book_data['bids'][0]['price']
        best_ask = book_data['asks'][0]['price']
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
    else:
        return None
    
    return {
        'order_book': book_data,
        'daily_volume': daily_volume if daily_volume > 0 else 100000,  # Default fallback
        'mid_price': mid_price,
        'spread': spread,
        'best_bid': best_bid,
        'best_ask': best_ask
    }


def predict_lob(order_book: Dict, time_elapsed: float, market_stats: Optional[Dict] = None) -> Dict:
    """
    Predict order book evolution through time.
    Pipeline: passive_recovery -> brownian_noise -> queue_reactive
    """
    book = copy.deepcopy(order_book)
    
    # Extract initial book from market_stats if available
    initial_book = market_stats.get('initial_book') if market_stats else None
    
    # Step 1: Passive recovery (Obizhaeva-Wang)
    book = passive_recovery(book, time_elapsed, initial_book)
    
    # Step 2: Brownian noise
    book = add_brownian_noise(book, time_elapsed)
    
    # Step 3: Queue reactive adjustment
    book = queue_reactive_adjustment(book, market_stats)
    
    return book



def passive_recovery(order_book: Dict, time_elapsed: float, initial_book: Dict = None) -> Dict:
    """
    Obizhaeva-Wang style liquidity regeneration using price buckets.
    Levels recover toward the average size of their price bucket from initial book.
    """
    if initial_book is None:
        return order_book
        
    book = copy.deepcopy(order_book)
    resilience_rate = 0.1  # Per second
    
    for side in ['bids', 'asks']:
        if not book[side] or not initial_book[side]:
            continue
            
        reference_price = initial_book[side][0]['price']
        
        # Build bucket profiles from initial book
        bucket_volumes = {}
        bucket_counts = {}
        
        for level in initial_book[side]:
            # Calculate bucket: exponential growth (0-10bps, 10-20bps, 20-40bps, etc)
            distance_bps = abs(level['price'] - reference_price) / reference_price * 10000
            if distance_bps < 10:
                bucket = 0
            else:
                bucket = int(math.log2(distance_bps / 10)) + 1
            
            if bucket not in bucket_volumes:
                bucket_volumes[bucket] = 0
                bucket_counts[bucket] = 0
            
            bucket_volumes[bucket] += level['amount']
            bucket_counts[bucket] += 1
        
        # Calculate equilibrium per bucket
        bucket_equilibrium = {}
        for bucket in bucket_volumes:
            bucket_equilibrium[bucket] = bucket_volumes[bucket] / bucket_counts[bucket]
        
        # Apply recovery to current book
        current_reference = book[side][0]['price'] if book[side] else reference_price
        
        for level in book[side]:
            # Find this level's bucket
            distance_bps = abs(level['price'] - current_reference) / current_reference * 10000
            if distance_bps < 10:
                bucket = 0
            else:
                bucket = int(math.log2(distance_bps / 10)) + 1
            
            # Get equilibrium size
            if bucket in bucket_equilibrium:
                equilibrium_size = bucket_equilibrium[bucket]
            elif bucket_equilibrium:
                # Use furthest bucket's size with decay
                max_bucket = max(bucket_equilibrium.keys())
                equilibrium_size = bucket_equilibrium[max_bucket] * (0.7 ** (bucket - max_bucket))
            else:
                equilibrium_size = level['amount']  # No change if no reference
            
            # Recover toward equilibrium
            if level['amount'] < equilibrium_size:
                deficit = equilibrium_size - level['amount']
                recovery = deficit * (1 - math.exp(-resilience_rate * time_elapsed))
                level['amount'] += recovery
    
    return book


def add_brownian_noise(order_book: Dict, time_elapsed: float) -> Dict:
    """
    Add random liquidity fluctuations.
    Models random order arrivals/cancellations.
    """
    book = copy.deepcopy(order_book)
    volatility = 0.2  # 20% volatility in liquidity
    
    for side in ['bids', 'asks']:
        for level in book[side]:
            # Brownian motion: dL = σ * sqrt(dt) * dW
            std_dev = level['amount'] * volatility * math.sqrt(time_elapsed)
            noise = np.random.normal(0, std_dev)
            # Ensure non-negative
            level['amount'] = max(0.1, level['amount'] + noise)
    
    return book


def queue_reactive_adjustment(order_book: Dict, market_stats: Optional[Dict] = None) -> Dict:
    """
    Adjust liquidity based on order book imbalance.
    Thin side attracts more liquidity.
    """
    book = copy.deepcopy(order_book)
    
    # Calculate imbalance
    total_bid_size = sum(level['amount'] for level in book['bids'][:5])
    total_ask_size = sum(level['amount'] for level in book['asks'][:5])
    
    if total_bid_size + total_ask_size > 0:
        imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
    else:
        imbalance = 0
    
    # Adjust liquidity: thin side gets boost
    boost_factor = 0.1  # 10% max adjustment
    
    if imbalance > 0.2:  # Bids are stronger, boost asks
        for level in book['asks'][:3]:  # Top 3 levels
            level['amount'] *= (1 + boost_factor)
    elif imbalance < -0.2:  # Asks are stronger, boost bids
        for level in book['bids'][:3]:  # Top 3 levels
            level['amount'] *= (1 + boost_factor)
    
    return book


def calculate_price_impact(volume: float, time_interval: float, daily_volume: float) -> Dict[str, float]:
    """
    Almgren-Chriss price impact model.
    Returns permanent and temporary impact.
    """
    # Convert daily volume to interval volume
    seconds_per_day = 86400
    interval_volume = daily_volume * (time_interval / seconds_per_day)
    
    # Model parameters (typical institutional values)
    gamma = 0.1  # Permanent impact constant
    eta = 0.5    # Temporary impact constant
    
    # Normalize volume
    if interval_volume > 0:
        volume_ratio = volume / interval_volume
    else:
        volume_ratio = volume / (daily_volume / (seconds_per_day / time_interval))
    
    # Permanent impact: linear in volume
    permanent = gamma * volume_ratio
    
    # Temporary impact: square-root for large orders
    temporary = eta * math.sqrt(volume_ratio)
    
    return {
        'permanent': permanent,
        'temporary': temporary,
        'total': permanent + temporary
    }


def calculate_participation_rate(aggressiveness: float, spread_fraction: float) -> float:
    """
    Calculate what fraction of visible liquidity we can capture.
    
    Args:
        aggressiveness: 0 = passive, 1 = aggressive, >1 = urgent
        spread_fraction: How far through spread our limit reaches (0 to 1+)
    """
    # Base participation depends on position in spread
    if spread_fraction < 0.1:  # Very passive
        base_rate = 0.2
    elif spread_fraction < 0.5:  # Near mid
        base_rate = 0.3 + 0.2 * (spread_fraction / 0.5)
    elif spread_fraction < 1.0:  # Up to touch
        base_rate = 0.5 + 0.2 * ((spread_fraction - 0.5) / 0.5)
    else:  # Through spread
        base_rate = 0.7 + 0.2 * min(1, spread_fraction - 1)
    
    # Aggressiveness adds urgency factor
    urgency_boost = aggressiveness * 0.1
    
    return min(0.95, base_rate + urgency_boost)


def calculate_limit_price(mid_price: float, spread: float, side: str, aggressiveness: float) -> float:
    """
    Calculate limit price for interval order based on aggressiveness.
    
    Args:
        aggressiveness: 0 = mid, 0.5 = touch, 1 = through spread
    """
    if side == 'buy':
        # Start at mid, move toward and through ask
        limit_price = mid_price + aggressiveness * spread
    else:
        # Start at mid, move toward and through bid
        limit_price = mid_price - aggressiveness * spread
    
    return limit_price


def calculate_execution_cost(
    symbol: str,
    side: str,
    volume: float,
    time_seconds: int,
    limit_price: Optional[float] = None
) -> Dict:
    """
    Master function: Estimate TWAP execution cost.
    """
    # 1. Setup
    interval_seconds = 10
    num_intervals = max(1, time_seconds // interval_seconds)
    base_volume_per_interval = volume / num_intervals
    
    # 2. Get market data
    market_data = get_market_data(symbol, use_live=False)
    if not market_data:
        return {'error': 'Failed to fetch market data'}
    
    # Save initial book as equilibrium reference
    initial_book = copy.deepcopy(market_data['order_book'])
    
    # Update market_stats to include initial book
    market_stats = {
        **market_data,
        'initial_book': initial_book
    }
    
    initial_mid = market_data['mid_price']
    spread = market_data['spread']
    daily_volume = market_data['daily_volume']
    
    # 3. Initialize tracking
    remaining_volume = volume
    current_book = copy.deepcopy(market_data['order_book'])
    cumulative_permanent_impact = 0
    aggressiveness = 0.0
    
    # Results tracking
    total_cost = 0
    executed_volume = 0
    execution_prices = []
    interval_fills = []
    
    # 4. Execute each interval
    for interval in range(num_intervals):
        if remaining_volume <= 0:
            break
        
        # Calculate target for this interval (with catch-up)
        intervals_left = num_intervals - interval
        interval_target = remaining_volume / intervals_left
        
        # Evolve order book
        if interval > 0:
            current_book = predict_lob(current_book, interval_seconds, market_stats)
        
        # Calculate current mid with permanent impact
        if current_book['bids'] and current_book['asks']:
            book_mid = (current_book['bids'][0]['price'] + current_book['asks'][0]['price']) / 2
            current_mid = book_mid + cumulative_permanent_impact
        else:
            current_mid = initial_mid + cumulative_permanent_impact
        
        # Determine limit price for this interval
        interval_limit = calculate_limit_price(current_mid, spread, side, aggressiveness)
        
        # Apply user limit if specified
        if limit_price:
            if side == 'buy' and interval_limit > limit_price:
                interval_limit = limit_price
            elif side == 'sell' and interval_limit < limit_price:
                interval_limit = limit_price
        
        # Calculate spread fraction (for participation)
        if side == 'buy':
            spread_fraction = (interval_limit - current_mid) / spread
        else:
            spread_fraction = (current_mid - interval_limit) / spread
        
        # Calculate participation rate
        participation = calculate_participation_rate(aggressiveness, spread_fraction)
        
        # Estimate fill from visible book
        available_liquidity = 0
        if side == 'buy':
            for level in current_book['asks']:
                if level['price'] <= interval_limit:
                    # Take participation % of this level
                    available_liquidity += level['amount'] * participation
        else:
            for level in current_book['bids']:
                if level['price'] >= interval_limit:
                    # Take participation % of this level
                    available_liquidity += level['amount'] * participation
        
        # Actual fill is minimum of target and available
        actual_fill = min(interval_target, available_liquidity, remaining_volume)
        
        if actual_fill > 0:
            # Calculate price impact
            impact = calculate_price_impact(actual_fill, interval_seconds, daily_volume)
            
            # Execution price includes temporary impact
            if side == 'buy':
                execution_price = interval_limit + impact['temporary'] * spread
            else:
                execution_price = interval_limit - impact['temporary'] * spread
            
            # Update cumulative permanent impact
            if side == 'buy':
                cumulative_permanent_impact += impact['permanent'] * spread
            else:
                cumulative_permanent_impact -= impact['permanent'] * spread
            
            # Execute trade
            total_cost += actual_fill * execution_price
            executed_volume += actual_fill
            remaining_volume -= actual_fill
            execution_prices.append(execution_price)
            
            # Deplete order book
            fill_remaining = actual_fill / participation  # Total depletion
            if side == 'buy':
                for level in current_book['asks']:
                    if level['price'] <= interval_limit and fill_remaining > 0:
                        taken = min(level['amount'], fill_remaining)
                        level['amount'] -= taken
                        fill_remaining -= taken
            else:
                for level in current_book['bids']:
                    if level['price'] >= interval_limit and fill_remaining > 0:
                        taken = min(level['amount'], fill_remaining)
                        level['amount'] -= taken
                        fill_remaining -= taken
        
        # Track interval performance
        interval_fills.append({
            'target': interval_target,
            'filled': actual_fill,
            'fill_rate': actual_fill / interval_target if interval_target > 0 else 0
        })
        
        # Adjust aggressiveness based on performance
        if actual_fill < interval_target * 0.95:  # Missed target
            # Increase aggressiveness proportionally to shortfall
            shortfall_ratio = 1 - (actual_fill / interval_target)
            aggressiveness = min(2.0, aggressiveness + shortfall_ratio * 0.5)
    
    # 5. Calculate final metrics
    avg_price = total_cost / executed_volume if executed_volume > 0 else 0
    fill_rate = (executed_volume / volume) * 100
    
    # Calculate slippage
    if side == 'buy':
        slippage_bps = ((avg_price - initial_mid) / initial_mid) * 10000
    else:
        slippage_bps = ((initial_mid - avg_price) / initial_mid) * 10000
    
    return {
        'symbol': symbol,
        'side': side,
        'requested_volume': volume,
        'executed_volume': executed_volume,
        'unfilled_volume': remaining_volume,
        'total_cost': total_cost,
        'avg_price': avg_price,
        'initial_mid_price': initial_mid,
        'final_aggressiveness': aggressiveness,
        'fill_rate': fill_rate,
        'slippage_bps': slippage_bps,
        'num_intervals': num_intervals,
        'execution_prices': execution_prices
    }


def main():
    """CLI interface for testing."""
    if len(sys.argv) < 5:
        print("\nUsage: python twap_clean.py <symbol> <side> <volume> <time_seconds> [limit_price]")
        print("\nRunning default test...")
        symbol = 'BTCUSD'
        side = 'buy'
        volume = 10
        time_seconds = 300  # 5 minutes
        limit_price = None
    else:
        symbol = sys.argv[1]
        side = sys.argv[2]
        volume = float(sys.argv[3])
        time_seconds = int(sys.argv[4])
        limit_price = float(sys.argv[5]) if len(sys.argv) > 5 else None
    
    print(f"\nEstimating TWAP execution:")
    print(f"Symbol: {symbol}")
    print(f"Side: {side}")
    print(f"Volume: {volume}")
    print(f"Time: {time_seconds}s ({time_seconds/60:.1f} min)")
    print(f"Limit: {limit_price if limit_price else 'None'}")
    
    result = calculate_execution_cost(symbol, side, volume, time_seconds, limit_price)
    
    print("\n=== RESULTS ===")
    for key, value in result.items():
        if isinstance(value, float):
            if 'price' in key or 'cost' in key:
                print(f"{key}: ${value:,.2f}")
            elif 'volume' in key:
                print(f"{key}: {value:.6f}")
            elif key == 'slippage_bps':
                print(f"{key}: {value:.1f} bps")
            elif key == 'fill_rate':
                print(f"{key}: {value:.1f}%")
            else:
                print(f"{key}: {value:.4f}")
        elif key == 'execution_prices':
            if value:
                print(f"price_range: ${min(value):,.2f} - ${max(value):,.2f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()