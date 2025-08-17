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
from symbols import REVERSED_TO_GEMINI, is_valid_pair, get_gemini_symbol

from symbols import SYMBOLS_SET

def split_pair(pair: str) -> tuple:
    """
    Split a pair string into two valid currency symbols.
    
    Args:
        pair: Trading pair string (e.g., 'BTCUSD', 'USDTBTC')
    
    Returns:
        Tuple of (base, quote) if valid, (None, None) if invalid
        
    Examples:
        'BTCUSD' -> ('BTC', 'USD')
        'USDTBTC' -> ('USDT', 'BTC') 
        'ABCDEF' -> (None, None)
    """
    pair_upper = pair.upper()
    
    # Try 3-character split first
    if len(pair_upper) >= 6:
        base_3 = pair_upper[:3]
        quote_3 = pair_upper[3:]
        
        if base_3 in SYMBOLS_SET and quote_3 in SYMBOLS_SET:
            return (base_3, quote_3)
    
    # Try 4-character split
    if len(pair_upper) >= 7:
        base_4 = pair_upper[:4]
        quote_4 = pair_upper[4:]
        
        if base_4 in SYMBOLS_SET and quote_4 in SYMBOLS_SET:
            return (base_4, quote_4)
    
    return (None, None)

def get_market_data(raw_symbol: str, use_live: bool = True) -> Dict:
    """
    Fetch order book and market statistics.
    """
    # Quick validity check first
    if not is_valid_pair(raw_symbol):
        return None
    
    # Get the proper Gemini symbol and check if it needs reversal
    symbol = get_gemini_symbol(raw_symbol)
    if not symbol:
        return None
    
    # Check if this is a reversed notation
    raw_lower = raw_symbol.lower()
    forward = raw_lower not in REVERSED_TO_GEMINI

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
        
        # If using reversed symbol, need to invert prices and swap sides
        if not forward:
            # Swap bids and asks (buying USD means selling BTC)
            original_bids = book_data['bids']
            original_asks = book_data['asks']
            
            # Convert asks to bids (and vice versa) with inverted prices
            book_data['bids'] = []
            for level in original_asks:
                original_price = level['price']
                book_data['bids'].append({
                    'price': 1 / original_price,
                    'amount': level['amount'] * original_price  # Convert to reversed denomination
                })
            
            book_data['asks'] = []
            for level in original_bids:
                original_price = level['price']
                book_data['asks'].append({
                    'price': 1 / original_price,
                    'amount': level['amount'] * original_price  # Convert to reversed denomination
                })
            
            # Sort properly (bids descending, asks ascending)
            book_data['bids'] = sorted(book_data['bids'], key=lambda x: -x['price'])
            book_data['asks'] = sorted(book_data['asks'], key=lambda x: x['price'])
                
    else:
        # Use cached data
        os.makedirs('test_data', exist_ok=True)
        filename = f'test_data/{raw_symbol}_snapshot.json'
        
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                book_data = json.load(f)
        else:
            # Fetch once and save - handle the None case!
            result = get_market_data(raw_symbol, use_live=True)
            if result is None:
                print(f"Failed to fetch data for {raw_symbol}")
                return None
            book_data = result['order_book']
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
        if forward:
            daily_volume = float(ticker['volume'][volume_keys[0]])
        else:
            # For reversed pairs, need to get the USD volume
            # volume_keys[0] is usually the base currency (e.g., BTC)
            # volume_keys[1] is usually USD
            # For reversed, we want the USD volume
            daily_volume = float(ticker['volume'][volume_keys[1]])
    except:
        daily_volume = 0
    
    # Calculate metrics
    if book_data and book_data['bids'] and book_data['asks']:
        best_bid = book_data['bids'][0]['price']
        best_ask = book_data['asks'][0]['price']
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
    else:
        return None
    
    return {
        'order_book': book_data,
        'daily_volume': daily_volume,
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
    
    # Extract pre-calculated bucket volumes if available
    initial_bucket_volumes = market_stats.get('initial_bucket_volumes') if market_stats else None
    
    # Step 1: Passive recovery (Obizhaeva-Wang)
    book = passive_recovery(book, time_elapsed, initial_bucket_volumes)
    
    # Step 2: Brownian noise
    book = add_brownian_noise(book, time_elapsed, volatility=0.01)
    
    # Step 3: Queue reactive adjustment
    book = queue_reactive_adjustment(book, market_stats)
    
    return book

def passive_recovery(order_book: Dict, time_elapsed: float, market_stats: Optional[Dict] = None) -> Dict:
    """
    Add new liquidity at competitive prices based on natural flow rate.
    """
    book = copy.deepcopy(order_book)
    
    if not market_stats or 'daily_volume' not in market_stats:
        return book
    
    # Natural flow rate from daily volume
    daily_volume = market_stats['daily_volume']
    flow_rate_per_second = daily_volume / 86400
    expected_new_liquidity = flow_rate_per_second * time_elapsed
    
    # Add new liquidity near current best prices
    for side in ['bids', 'asks']:
        if not book[side]:
            continue
            
        # Get current best price
        best_price = book[side][0]['price']
        
        # Add competitive liquidity at and beyond best price
        # Distribution: 40% at touch, 30% next tick, 20% next, 10% next
        distribution = [0.4, 0.3, 0.2, 0.1]
        tick_size = 0.01
        
        for i, fraction in enumerate(distribution):
            # Calculate price for new liquidity
            if side == 'asks':
                new_price = best_price + i * tick_size
            else:  # bids
                new_price = best_price - i * tick_size
            
            # Amount for this level
            new_amount = (expected_new_liquidity / 2) * fraction
            
            # Add to existing level or create new
            level_exists = False
            for level in book[side]:
                if abs(level['price'] - new_price) < tick_size / 2:
                    level['amount'] += new_amount
                    level_exists = True
                    break
            
            if not level_exists:
                book[side].append({'price': new_price, 'amount': new_amount})
        
        # Sort and trim
        if side == 'asks':
            book[side] = sorted(book[side], key=lambda x: x['price'])[:50]
        else:
            book[side] = sorted(book[side], key=lambda x: -x['price'])[:50]
    
    return book

def calculate_bucket_volumes(order_book: Dict) -> Dict:
    """
    Calculate total volume per price bucket for each side of the book.
    Used as equilibrium reference for passive recovery.
    """
    bucket_volumes = {'bids': {}, 'asks': {}}
    
    for side in ['bids', 'asks']:
        if not order_book[side]:
            continue
            
        reference_price = order_book[side][0]['price']
        
        for level in order_book[side]:
            # Calculate bucket: exponential growth (0-10bps, 10-20bps, 20-40bps, etc)
            distance_bps = abs(level['price'] - reference_price) / reference_price * 10000
            if distance_bps < 10:
                bucket = 0
            else:
                bucket = int(math.log2(distance_bps / 10)) + 1
            
            if bucket not in bucket_volumes[side]:
                bucket_volumes[side][bucket] = 0
            
            bucket_volumes[side][bucket] += level['amount']
    
    return bucket_volumes

def add_brownian_noise(order_book: Dict, time_elapsed: float, volatility: float) -> Dict:
    """
    Add random liquidity fluctuations.
    Models zero-intelligence market activity using Brownian motion.
    """
    book = copy.deepcopy(order_book)
    
    for side in ['bids', 'asks']:
        for level in book[side]:
            # Brownian motion: dL = σ * sqrt(dt) * dW
            std_dev = level['amount'] * volatility * math.sqrt(time_elapsed)
            noise = np.random.normal(0, std_dev)
            # Ensure non-negative
            level['amount'] = max(0, level['amount'] + noise)
    
    return book

def queue_reactive_adjustment(order_book: Dict, market_stats: Optional[Dict] = None) -> Dict:
    """
    TODO: Improve queue-reactive model.
    Simple queue-reactive model: thin side attracts liquidity, thick side loses it.
    Based on the core insight that traders prefer less crowded queues.
    """
    book = copy.deepcopy(order_book)
    
    if not book['bids'] or not book['asks']:
        return book
    
    # Calculate volume imbalance (within 5 levels)
    bid_volume = sum(level['amount'] for level in book['bids'][:5])
    ask_volume = sum(level['amount'] for level in book['asks'][:5])
    total_volume = bid_volume + ask_volume
    
    if total_volume == 0:
        return book
    
    # Imbalance: -1 (all asks) to +1 (all bids)
    imbalance = (bid_volume - ask_volume) / total_volume
    
    # Base flow rate: 10% of average level size
    avg_level_size = total_volume / 10
    base_flow = avg_level_size * 0.1
    
    # Adjust each side based on imbalance
    # Thin side gains liquidity, thick side loses it
    for side in ['bids', 'asks']:
        # Bid side: positive imbalance means we're thick (lose liquidity)
        # Ask side: negative imbalance means we're thick (lose liquidity)
        side_imbalance = imbalance if side == 'bids' else -imbalance
        
        for i, level in enumerate(book[side][:5]):
            if side_imbalance > 0.2:  # We're the thick side
                # Orders cancel/migrate away
                outflow = base_flow * side_imbalance * (1 - i/5)  # More at top
                level['amount'] = max(0.1, level['amount'] - outflow)
                
            elif side_imbalance < -0.2:  # We're the thin side  
                # New orders arrive
                inflow = base_flow * abs(side_imbalance) * (1 - i/5)  # More at top
                level['amount'] += inflow
    
    return book

def calculate_price_impact(volume: float, daily_volume: float, price: float) -> Dict[str, float]:
    """
    Models only permanent impact using Almgren-Chriss.
    Temporary impact is already captured by walking up the LOB.
    """
    daily_volume_pct = volume / daily_volume if daily_volume > 0 else 0.01
    interval_flow = daily_volume * (10 / 86400)
    participation_rate = volume / interval_flow if interval_flow > 0 else 1.0
    
    # Only permanent impact - shifts the mid-price for future intervals
    gamma = participation_rate * 10 # gamma increases by 1 for every 10% participation rate
    permanent_bps = gamma * daily_volume_pct * 100
    permanent_price = (permanent_bps / 10000) * price
    
    return {
        'permanent': permanent_price,
        'temporary': 0,  # Already captured by LOB walking
        'total': permanent_price
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

def calculate_transaction_cost(order_volume: float, daily_volume: float, monthly_volume: float) -> float:
    """Calculate transaction cost based on maker/taker fee percentages for Gemini"""
    fees = [
        (100_000_000, 0.00, 0.04),
        (50_000_000, 0.00, 0.05),
        (10_000_000, 0.02, 0.08),
        (5_000_000, 0.03, 0.10),
        (1_000_000, 0.05, 0.15),
        (100_000, 0.08, 0.20),
        (50_000, 0.10, 0.25),
        (10_000, 0.15, 0.30),
        (0, 0.20, 0.40)
    ] # from Gemini
    
    for threshold, maker, taker in fees:
        if monthly_volume >= threshold:
            tx_rate = (maker/100, taker/100) 
    
    return order_volume * tx_rate[0] # TODO: use order_volume and daily_volume to find maker/taker split

def calculate_execution_cost(
    symbol: str,
    side: str,
    volume: float,
    time_seconds: int,
    limit_price: Optional[float] = None,
    use_live: bool = True
) -> Dict:
    """
    Master function: Estimate TWAP execution cost.
    """

    side = side.lower()
    if side not in ['buy', 'sell']:
        return {'error': 'Invalid side. Use "buy" or "sell".'}

    # 1. Setup
    interval_seconds = 10
    num_intervals = max(1, time_seconds // interval_seconds)
    base_volume_per_interval = volume / num_intervals
    
    # 2. Get market data
    market_data = get_market_data(symbol, use_live)
    if not market_data:
        return {'error': 'Failed to fetch market data'}
    
    # Calculate initial bucket volumes for passive recovery
    initial_book = copy.deepcopy(market_data['order_book'])
    initial_bucket_volumes = calculate_bucket_volumes(initial_book)
    
    # Update market_stats to include bucket volumes instead of full book
    market_stats = {
        **market_data,
        'initial_bucket_volumes': initial_bucket_volumes 
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
            impact = calculate_price_impact(actual_fill, daily_volume, current_mid)
    
            # Execution price includes temporary impact
            if side == 'buy':
                execution_price = interval_limit + impact['temporary']
            else:
                execution_price = interval_limit - impact['temporary']
            
            # Update cumulative permanent impact
            if side == 'buy':
                cumulative_permanent_impact += impact['permanent']
            else:
                cumulative_permanent_impact -= impact['permanent']

            # After calculating permanent impact, shift the entire order book
            if side == 'buy':  # If we're buying, prices go up
                for level in current_book['bids']:
                    level['price'] += impact['permanent']
                for level in current_book['asks']:
                    level['price'] += impact['permanent']
            else:  # If we're selling, prices go down
                for level in current_book['bids']:
                    level['price'] -= impact['permanent']
                for level in current_book['asks']:
                    level['price'] -= impact['permanent']
            
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
    """
    CLI interface for testing.
    Usage: python twap_clean.py <symbol> <side> <volume> <time_seconds> [limit_price]
    """
    if len(sys.argv) < 5:
        print("\nRunning default test...")
        symbol = 'USDBTC'
        side = 'sell'
        volume = 1000000
        time_seconds = 6*60*60
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
    
    result = calculate_execution_cost(symbol, side, volume, time_seconds, limit_price, use_live=False)
    
    print("\n=== RESULTS ===")
    for key, value in result.items():
        if isinstance(value, float):
            if 'price' in key or 'cost' in key:
                # Use scientific notation for very small/large numbers, otherwise fixed
                if value != 0 and (abs(value) < 0.01 or abs(value) > 1000000):
                    print(f"{key}: {value:.8e}")
                else:
                    print(f"{key}: {value:.8f}")
            elif 'volume' in key:
                print(f"{key}: {value:.8f}")
            elif key == 'slippage_bps':
                print(f"{key}: {value:.2f} bps")
            elif key == 'fill_rate':
                print(f"{key}: {value:.2f}%")
            elif key == 'aggressiveness':
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value:.6f}")
        elif key == 'execution_prices':
            if value:
                min_price = min(value)
                max_price = max(value)
                # Use appropriate formatting based on magnitude
                if min_price != 0 and (abs(min_price) < 0.01 or abs(min_price) > 1000000):
                    print(f"price_range: {min_price:.8e} - {max_price:.8e}")
                else:
                    print(f"price_range: {min_price:.8f} - {max_price:.8f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()