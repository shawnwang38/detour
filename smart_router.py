#!/usr/bin/env python3
"""
TO BE UPDATED DO NOT RUN
Smart Order Routing Algorithm
Finds optimal route combinations for crypto execution
"""

# TODO: implement reverse execution
from twap_estimator import estimate_twap_execution, estimate_reverse_twap_execution
import numpy as np
from typing import Dict, List, Tuple, Optional

# Supported markets (expandable via CSV)
SUPPORTED_PAIRS = ['LINKUSD', 'LINKBTC', 'LINKETH', 'BTCUSD', 'ETHUSD', 'ETHBTC', 'USDTUSD', 'USDCUSD']

# Extract unique symbols from supported pairs
def extract_symbols_from_pairs(pairs: List[str]) -> set:
    """Extract all unique 3-4 letter symbols from pair list"""
    symbols = set()
    
    # Known 4-letter symbols
    four_letter = {'USDT', 'USDC', 'GUSD', 'LINK', 'AVAX', 'DOGE', 'MATIC', 'ATOM'}
    
    for pair in pairs:
        # Try 4-letter splits first
        found = False
        for symbol in four_letter:
            if pair.startswith(symbol):
                symbols.add(symbol)
                symbols.add(pair[len(symbol):])
                found = True
                break
            elif pair.endswith(symbol):
                symbols.add(pair[:-len(symbol)])
                symbols.add(symbol)
                found = True
                break
        
        # Fallback to 3-letter split
        if not found and len(pair) == 6:
            symbols.add(pair[:3])
            symbols.add(pair[3:])
    
    return symbols

SUPPORTED_SYMBOLS = extract_symbols_from_pairs(SUPPORTED_PAIRS)

def split_pair(pair: str) -> Tuple[str, str]:
    """Split a pair like LINKUSD into (LINK, USD)"""
    # Known 4-letter symbols
    four_letter = {'USDT', 'USDC', 'GUSD', 'LINK', 'AVAX', 'DOGE', 'MATIC', 'ATOM'}
    
    # Check 4-letter prefixes
    for symbol in four_letter:
        if pair.startswith(symbol):
            return symbol, pair[len(symbol):]
        elif pair.endswith(symbol):
            return pair[:-len(symbol)], symbol
    
    # Default to 3-letter split
    if len(pair) == 6:
        return pair[:3], pair[3:]
    
    raise ValueError(f"Cannot split pair: {pair}")

def find_market(base: str, quote: str) -> Optional[str]:
    """Find if a market exists in our supported pairs"""
    direct = base + quote
    reverse = quote + base
    
    if direct in SUPPORTED_PAIRS:
        return direct
    elif reverse in SUPPORTED_PAIRS:
        return reverse
    return None

def get_execution_side(base: str, quote: str, target: str) -> str:
    """Determine if we're buying or selling in a market"""
    market = find_market(base, quote)
    if not market:
        return None
    
    # If market is BASEQUOTE and we want QUOTE, we're selling BASE (ask side)
    # If market is QUOTEBASE and we want BASE, we're buying QUOTE (bid side)
    if market == base + quote:
        return 'ask' if target == quote else 'bid'
    else:  # market == quote + base
        return 'bid' if target == quote else 'ask'

def get_fee_tier(monthly_volume: float) -> Tuple[float, float]:
    """Get maker/taker fee percentages based on 30-day volume"""
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
    ]
    
    for threshold, maker, taker in fees:
        if monthly_volume >= threshold:
            return (maker/100, taker/100)  # Convert to decimal
    
    return (0.20/100, 0.40/100)

def evaluate_route_combination(routes: List[dict], allocation: List[float], 
                             total_volume: float, total_time: int,
                             monthly_volume: float, cache: dict,
                             direct_route_info: Dict = None,
                             verbose: bool = False) -> Tuple[float, Dict]:
    """Evaluate total cost of a route combination with allocations"""
    cache_key = tuple(allocation)
    if cache_key in cache:
        return cache[cache_key]
    
    total_cost = 0
    total_filled = 0
    total_execution_cost = 0
    total_tx_fees = 0
    route_details = []
    
    maker_fee, taker_fee = get_fee_tier(monthly_volume)
    avg_fee = 0.3 * maker_fee + 0.7 * taker_fee
    
    for route_idx, percent in enumerate(allocation):
        if percent == 0:
            continue
            
        route = routes[route_idx]
        volume = total_volume * percent / 100
        
        route_cost = 0
        route_execution_cost = 0
        route_filled_volume = volume
        route_avg_price = 0
        route_tx_fees = 0
        
        if route['stops'] == 0:
            # Direct route
            segment = route['segments'][0]
            result = estimate_twap_execution(
                segment['pair'], segment['side'], volume, total_time
            )
            route_execution_cost = result['total_cost']
            route_filled_volume = result['executed_volume']
            route_avg_price = result['avg_price']
            
            # Simple fee calculation for direct route
            route_tx_fees = route_execution_cost * avg_fee
            
        else:
            # Multi-segment route
            if route.get('is_bid', False):
                # Bid detour: Execute backwards
                segments = route['segments'][::-1]
                
                # Last segment (leg 2): Buy target volume of base with intermediate
                last_seg = segments[0]
                result2 = estimate_twap_execution(
                    last_seg['pair'], last_seg['side'], volume, int(total_time / 2)
                )
                
                intermediate_needed = result2['total_cost']
                route_filled_volume = result2['executed_volume']
                
                # First segment (leg 1): Use reverse execution to get intermediate
                first_seg = segments[1]
                reverse_result = estimate_reverse_twap_execution(
                    first_seg['pair'], first_seg['side'], intermediate_needed, int(total_time / 2)
                )
                
                if 'error' in reverse_result:
                    # Fallback: approximate
                    route_execution_cost = intermediate_needed * 1.1  # Add 10% buffer
                    route_tx_fees = route_execution_cost * avg_fee * 1.5
                else:
                    route_execution_cost = reverse_result['base_volume_needed']
                    result1 = reverse_result['execution_result']
                    
                    # Calculate fees for both legs
                    leg1_fee = route_execution_cost * avg_fee
                    leg2_notional = result2['executed_volume'] * result2['avg_price']
                    leg2_fee_native = leg2_notional * avg_fee
                    
                    # Convert leg2 fee to quote currency (approximate)
                    if first_seg['pair'] in ['ETHUSD', 'ETHBTC'] and result1['avg_price'] > 0:
                        leg2_fee = leg2_fee_native / result1['avg_price']
                    else:
                        leg2_fee = leg2_fee_native * 0.0003  # Rough approximation
                    
                    route_tx_fees = leg1_fee + leg2_fee
                
                # Average price is quote per base
                route_avg_price = route_execution_cost / route_filled_volume if route_filled_volume > 0 else 0
                
            else:
                # Ask detour: Execute forward
                remaining_volume = volume
                route_execution_cost = volume  # We're selling this much base
                
                # Execute segments forward
                for i, segment in enumerate(route['segments']):
                    seg_time = int(total_time / len(route['segments']))
                    result = estimate_twap_execution(
                        segment['pair'], segment['side'], remaining_volume, seg_time
                    )
                    
                    if i == 0:
                        # First segment tells us fill rate
                        actual_base_sold = result['executed_volume']
                        route_execution_cost = actual_base_sold
                    
                    remaining_volume = result['total_cost']
                
                route_filled_volume = remaining_volume  # Final quote received
                route_avg_price = route_filled_volume / route_execution_cost if route_execution_cost > 0 else 0
                
                # Approximate fees for 2-leg ask
                route_tx_fees = route_filled_volume * avg_fee * 1.5
        
        route_cost = route_execution_cost + route_tx_fees
        
        if verbose:
            filled_pct = route_filled_volume / volume * 100 if volume > 0 else 0
            print(f"  Route {route['name']}: vol={volume:.2f}, filled={route_filled_volume:.2f} ({filled_pct:.1f}%), "
                  f"exec_cost={route_execution_cost:.4f}, fees={route_tx_fees:.4f}, total={route_cost:.4f}")
        
        total_cost += route_cost
        total_execution_cost += route_execution_cost
        total_tx_fees += route_tx_fees
        total_filled += route_filled_volume
        
        route_details.append({
            'route': route['name'],
            'allocation': percent,
            'volume': volume,
            'filled': route_filled_volume,
            'avg_price': route_avg_price,
            'execution_cost': route_execution_cost,
            'tx_fees': route_tx_fees,
            'total_cost': route_cost
        })
    
    fill_rate = total_filled / total_volume * 100 if total_volume > 0 else 0
    
    details = {
        'total_cost': total_cost,
        'execution_cost': total_execution_cost,
        'tx_fees': total_tx_fees,
        'fill_rate': fill_rate,
        'routes': route_details
    }
    
    result = (total_cost, details)
    cache[cache_key] = result
    
    return result

def generate_smart_grid(n: int) -> List[List[float]]:
    """Generate strategic test points for n routes"""
    if n == 1:
        return [[100]]
    elif n == 2:
        return [[100, 0], [80, 20], [60, 40], [50, 50], [40, 60], [20, 80], [0, 100]]
    elif n == 3:
        return [
            [100, 0, 0], [0, 100, 0], [0, 0, 100],
            [70, 30, 0], [70, 0, 30], [0, 70, 30],
            [50, 50, 0], [50, 0, 50], [0, 50, 50],
            [60, 20, 20], [20, 60, 20], [20, 20, 60],
            [40, 30, 30], [33, 33, 34]
        ]
    else:
        points = []
        # Corners
        for i in range(n):
            point = [0.0] * n
            point[i] = 100
            points.append(point)
        # Some edges
        for i in range(min(n, 3)):
            for j in range(i+1, min(n, 3)):
                point = [0.0] * n
                point[i] = 50
                point[j] = 50
                points.append(point)
        # Equal split
        points.append([100/n] * n)
        return points

def smart_order_route(pair: str, side: str, volume: float, time_seconds: int,
                     gemini_monthly_volume: Optional[float] = None,
                     refinement_level: str = 'normal') -> Dict:
    """
    Main smart order routing algorithm
    """
    print(f"\n{'='*80}")
    print(f"Smart Order Routing: {volume} {pair} {side.upper()} over {time_seconds}s")
    print(f"{'='*80}")
    
    # Estimate monthly volume if not provided
    if gemini_monthly_volume is None:
        gemini_monthly_volume = volume * 2
    
    maker_fee, taker_fee = get_fee_tier(gemini_monthly_volume)
    print(f"Monthly volume: ${gemini_monthly_volume:,.2f} → Fees: {maker_fee*100:.2f}%/{taker_fee*100:.2f}% (maker/taker)")
    
    # Split pair
    base, quote = split_pair(pair)
    target = quote if side == 'ask' else base
    
    # Step 1: Test direct route
    print(f"\n{'='*60}")
    print("STEP 1: Testing Direct Route")
    print(f"{'='*60}")
    direct_result = estimate_twap_execution(pair, side, volume, time_seconds)
    
    print(f"Direct route results:")
    print(f"  Fill rate: {direct_result['fill_rate']:.1f}%")
    print(f"  Avg price: ${direct_result['avg_price']:.6f}")
    print(f"  Slippage: {direct_result['slippage_bps']:.2f} bps")
    print(f"  Execution cost: ${direct_result['total_cost']:.2f}")
    
    # Add tx fees for direct route
    direct_notional = direct_result['executed_volume'] * direct_result['avg_price']
    direct_tx_fee = direct_notional * (0.3 * maker_fee + 0.7 * taker_fee)
    direct_total_cost = direct_result['total_cost'] + direct_tx_fee
    print(f"  Tx fees: ${direct_tx_fee:.2f}")
    print(f"  Total cost: ${direct_total_cost:.2f}")
    
    # Check if direct route is good enough
    if direct_result['fill_rate'] == 100 and direct_result['slippage_bps'] < 5:
        print(f"\n✓ Direct route optimal: {direct_result['slippage_bps']:.2f} bps slippage")
        return {
            'routes': [pair],
            'allocation': [100],
            'total_cost': direct_total_cost,
            'avg_price': direct_result['avg_price'],
            'fill_rate': 100
        }
    
    # Warn if direct route has poor fill
    if direct_result['fill_rate'] < 100:
        print(f"\n⚠️  WARNING: Direct route only fills {direct_result['fill_rate']:.1f}% of requested volume!")
        print(f"   Will explore detour routes to improve fill rate.")
    
    # Step 2: Calculate detour allocation
    if direct_result['fill_rate'] < 100:
        detour_allocation = 100 - direct_result['fill_rate']
        print(f"\nDirect route can only fill {direct_result['fill_rate']:.1f}%, need detours for remaining {detour_allocation:.1f}%")
    else:
        detour_allocation = 20
        print(f"\nDirect route fills 100% but has high slippage ({direct_result['slippage_bps']:.0f} bps)")
        print(f"Testing {detour_allocation}% allocation to detours for potential cost savings")
    
    direct_allocation = min(direct_result['fill_rate'], 100 - detour_allocation)
    
    # Get more accurate price estimate at the actual direct allocation volume
    if direct_allocation > 0 and direct_allocation < 100:
        direct_partial = estimate_twap_execution(
            pair, side, volume * direct_allocation / 100, 
            int(time_seconds * direct_allocation / 100)
        )
        conversion_price = direct_partial['avg_price']
    else:
        conversion_price = direct_result['avg_price']
    
    print(f"\nInitial allocation plan: {direct_allocation:.1f}% direct, {detour_allocation:.1f}% detours")
    
    # Step 3: Find and test detours
    print(f"\n{'='*60}")
    print("STEP 2: Finding and Testing Detours")
    print(f"{'='*60}")
    
    shortlist = []
    tested_detours = []
    
    for intermediate in SUPPORTED_SYMBOLS:
        if intermediate in [base, quote]:
            continue
        
        detour_processed = False
        
        if side == 'bid':
            # BID: We want to buy 'base' with 'quote'
            test_target_volume = volume * detour_allocation / 100
            
            # Leg 2: How much intermediate to buy target base?
            leg2_market = find_market(intermediate, base)
            leg2_side = get_execution_side(intermediate, base, base) if leg2_market else None
            
            if not leg2_market or not leg2_side:
                print(f"\nSkipping {intermediate} detour - no market between {intermediate} and {base}")
                continue
                
            # Execute leg 2 with target volume
            leg2_result = estimate_twap_execution(
                leg2_market, leg2_side, test_target_volume, int(time_seconds // 2)
            )
            
            # How much intermediate currency do we need?
            intermediate_needed = leg2_result['total_cost']
            
            # Leg 1: How much quote to get that intermediate?
            leg1_market = find_market(quote, intermediate)
            leg1_side = get_execution_side(quote, intermediate, intermediate) if leg1_market else None
            
            if not leg1_market or not leg1_side:
                print(f"\nSkipping {intermediate} detour - no market between {quote} and {intermediate}")
                continue
            
            # For bid detours, we need reverse execution for leg 1
            leg1_reverse = estimate_reverse_twap_execution(
                leg1_market, leg1_side, intermediate_needed, int(time_seconds // 2)
            )
            if 'error' in leg1_reverse:
                print(f"\nTesting detour via {intermediate} - FAILED (reverse execution error)")
                continue
            leg1_result = leg1_reverse['execution_result']
            quote_needed = leg1_reverse['base_volume_needed']
            
            print(f"\nTesting detour via {intermediate}:")
            print(f"  Target: {test_target_volume:.2f} {base}")
            print(f"  Leg 2: {leg2_market} {leg2_side}")
            print(f"    Volume: {test_target_volume:.2f} {base}")
            print(f"    Fill rate: {leg2_result['fill_rate']:.1f}%")
            print(f"    Avg price: {leg2_result['avg_price']:.6f}")
            print(f"    Slippage: {leg2_result['slippage_bps']:.2f} bps")
            print(f"    Cost: {intermediate_needed:.2f} {intermediate}")
            print(f"  Leg 1: {leg1_market} {leg1_side} (reverse execution)")
            print(f"    Target output: {intermediate_needed:.2f} {intermediate}")
            print(f"    ETH needed: {quote_needed:.2f} {quote}")
            print(f"    Fill rate: {leg1_result['fill_rate']:.1f}%")
            print(f"    Avg price: {leg1_result['avg_price']:.6f}")
            print(f"    Slippage: {leg1_result['slippage_bps']:.2f} bps")
            
            if leg1_result['fill_rate'] > 50 and leg2_result['fill_rate'] > 50:
                detour_execution_cost = quote_needed
                final_volume_received = leg2_result['executed_volume']
                
                # Calculate tx fees
                avg_fee = 0.3 * maker_fee + 0.7 * taker_fee
                
                # Leg 1 fees
                leg1_fee_eth = quote_needed * avg_fee
                
                # Leg 2 fees
                leg2_notional = leg2_result['executed_volume'] * leg2_result['avg_price']
                leg2_fee_native = leg2_notional * avg_fee
                
                # Convert leg2 fee to ETH
                if intermediate == 'USD':
                    leg2_fee_eth = leg2_fee_native / leg1_result['avg_price'] if leg1_result['avg_price'] > 0 else 0
                elif intermediate == 'BTC':
                    leg2_fee_eth = leg2_fee_native / leg1_result['avg_price'] if leg1_result['avg_price'] > 0 else 0
                else:
                    leg2_fee_eth = 0
                
                print(f"\n  Transaction fees:")
                print(f"    Leg 1: {quote_needed:.2f} {quote} @ {avg_fee*100:.3f}% = {leg1_fee_eth:.4f} {quote}")
                print(f"    Leg 2: {leg2_notional:.2f} {intermediate} @ {avg_fee*100:.3f}% = {leg2_fee_native:.2f} {intermediate} = {leg2_fee_eth:.4f} {quote}")
                
                detour_tx_fee = leg1_fee_eth + leg2_fee_eth
                detour_total_cost = detour_execution_cost + detour_tx_fee
                
                print(f"\n  Summary:")
                print(f"    Final {base} received: {final_volume_received:.2f} (target was {test_target_volume:.2f})")
                print(f"    Total {quote} cost: {detour_execution_cost:.2f} {quote}")
                print(f"    Avg price per {base}: {detour_execution_cost / final_volume_received:.6f} {quote}/{base}")
                print(f"    Tx fees: {detour_tx_fee:.4f} {quote}")
                print(f"    Total cost: {detour_total_cost:.2f} {quote}")
                
                detour_cost_per_unit = detour_execution_cost / final_volume_received if final_volume_received > 0 else float('inf')
                direct_cost_per_unit = conversion_price
                
                print(f"\n  Cost per {base}:")
                print(f"    Detour: {detour_cost_per_unit:.6f} {quote}/{base}")
                print(f"    Direct: {direct_cost_per_unit:.6f} {quote}/{base}")
                print(f"    Detour is {abs((direct_cost_per_unit - detour_cost_per_unit) / direct_cost_per_unit * 100):.1f}% {'cheaper' if detour_cost_per_unit < direct_cost_per_unit else 'more expensive'}")
                
                detour_processed = True
            else:
                print(f"\n  ✗ Skipped: Low fill rates (Leg1: {leg1_result['fill_rate']:.1f}%, Leg2: {leg2_result['fill_rate']:.1f}%)")
                continue
                
        else:
            # ASK: We want to sell 'base' for 'quote'
            test_source_volume = volume * detour_allocation / 100
            
            # Leg 1: Sell base for intermediate
            leg1_market = find_market(base, intermediate)
            leg1_side = get_execution_side(base, intermediate, intermediate) if leg1_market else None
            
            # Leg 2: Sell intermediate for quote
            leg2_market = find_market(intermediate, quote)
            leg2_side = get_execution_side(intermediate, quote, quote) if leg2_market else None
            
            if not leg1_market or not leg2_market or not leg1_side or not leg2_side:
                if not leg1_market:
                    print(f"\nSkipping {intermediate} detour - no market between {base} and {intermediate}")
                else:
                    print(f"\nSkipping {intermediate} detour - no market between {intermediate} and {quote}")
                continue
                
            print(f"\nTesting detour via {intermediate}:")
            print(f"  Source: {test_source_volume:.4f} {base}")
            print(f"  Leg 1: {leg1_market} {leg1_side}")
            print(f"    Volume: {test_source_volume:.4f} {base}")
            
            # Execute first leg
            leg1_result = estimate_twap_execution(
                leg1_market, leg1_side, test_source_volume, int(time_seconds // 2)
            )
            
            print(f"    Fill rate: {leg1_result['fill_rate']:.1f}%")
            print(f"    Avg price: {leg1_result['avg_price']:.6f}")
            print(f"    Slippage: {leg1_result['slippage_bps']:.2f} bps")
            print(f"    Received: {leg1_result['total_cost']:.2f} {intermediate}")
            
            if leg1_result['fill_rate'] > 50:
                # Execute second leg with output from first
                print(f"  Leg 2: {leg2_market} {leg2_side}")
                print(f"    Volume: {leg1_result['total_cost']:.2f} {intermediate}")
                
                leg2_result = estimate_twap_execution(
                    leg2_market, leg2_side, leg1_result['total_cost'], int(time_seconds // 2)
                )
                
                quote_received = leg2_result['total_cost']
                print(f"    Fill rate: {leg2_result['fill_rate']:.1f}%")
                print(f"    Avg price: {leg2_result['avg_price']:.6f}")
                print(f"    Slippage: {leg2_result['slippage_bps']:.2f} bps")
                print(f"    Received: {quote_received:.2f} {quote}")
                
                if leg2_result['fill_rate'] > 50:
                    # Calculate tx fees
                    avg_fee = 0.3 * maker_fee + 0.7 * taker_fee
                    leg1_notional = leg1_result['executed_volume'] * leg1_result['avg_price']
                    leg2_notional = leg2_result['executed_volume'] * leg2_result['avg_price']
                    detour_tx_fee = (leg1_notional + leg2_notional) * avg_fee
                    
                    print(f"\n  Transaction fees:")
                    print(f"    Leg 1 notional: {leg1_notional:.2f} @ {avg_fee*100:.3f}% = {leg1_notional * avg_fee:.2f}")
                    print(f"    Leg 2 notional: {leg2_notional:.2f} @ {avg_fee*100:.3f}% = {leg2_notional * avg_fee:.2f}")
                    
                    # For ask orders, we care about net proceeds
                    net_proceeds = quote_received - detour_tx_fee
                    detour_execution_cost = test_source_volume
                    
                    print(f"\n  Summary:")
                    print(f"    Started with: {test_source_volume:.2f} {base}")
                    print(f"    Final {quote} received: {quote_received:.2f}")
                    print(f"    Tx fees: {detour_tx_fee:.2f}")
                    print(f"    Net proceeds: {net_proceeds:.2f} {quote}")
                    print(f"    Effective price per {base}: {net_proceeds / test_source_volume:.6f} {quote}/{base}")
                    
                    detour_revenue_per_unit = net_proceeds / test_source_volume if test_source_volume > 0 else 0
                    direct_revenue_per_unit = conversion_price
                    
                    print(f"\n  Revenue per {base}:")
                    print(f"    Detour: {detour_revenue_per_unit:.6f} {quote}/{base}")
                    print(f"    Direct: {direct_revenue_per_unit:.6f} {quote}/{base}")
                    print(f"    Detour is {abs((detour_revenue_per_unit - direct_revenue_per_unit) / direct_revenue_per_unit * 100):.1f}% {'better' if detour_revenue_per_unit > direct_revenue_per_unit else 'worse'}")
                    
                    detour_processed = True
                    detour_cost_per_unit = detour_revenue_per_unit
                    final_volume_received = quote_received
                    detour_total_cost = test_source_volume * conversion_price - net_proceeds
                else:
                    print(f"  ✗ Skipped: Second leg fill rate too low ({leg2_result['fill_rate']:.1f}%)")
                    continue
            else:
                print(f"  ✗ Skipped: First leg fill rate too low ({leg1_result['fill_rate']:.1f}%)")
                continue
        
        # Shortlisting logic
        if detour_processed:
            tested_detours.append({
                'intermediate': intermediate,
                'cost': detour_total_cost,
                'cost_per_unit': detour_cost_per_unit,
                'fill_rate': (final_volume_received / (test_target_volume if side == 'bid' else test_source_volume)) * 100
            })
            
            should_shortlist = False
            reason = ""
            
            # Priority 1: If direct route doesn't fill 100%, always consider alternatives
            if direct_result['fill_rate'] < 100:
                fill_improvement = ((final_volume_received / (test_target_volume if side == 'bid' else test_source_volume)) * 100) - direct_result['fill_rate']
                if fill_improvement > 5:  # At least 5% improvement
                    should_shortlist = True
                    reason = f"provides {fill_improvement:.1f}% better fill rate"
            # Priority 2: If detour is significantly cheaper/better per unit
            elif side == 'bid' and detour_cost_per_unit < direct_cost_per_unit * 0.95:
                should_shortlist = True
                reason = f"cheaper per unit ({detour_cost_per_unit:.6f} vs {direct_cost_per_unit:.6f})"
            elif side == 'ask' and detour_revenue_per_unit > direct_revenue_per_unit * 1.05:
                should_shortlist = True
                reason = f"better revenue per unit ({detour_revenue_per_unit:.6f} vs {direct_revenue_per_unit:.6f})"
                
            if should_shortlist:
                print(f"  ✓ Added to shortlist ({reason})")
                shortlist.append({
                    'intermediate': intermediate,
                    'segments': [
                        {'pair': leg1_market, 'side': leg1_side},
                        {'pair': leg2_market, 'side': leg2_side}
                    ],
                    'test_cost': detour_total_cost,
                    'cost_per_unit': detour_cost_per_unit,
                    'is_bid': side == 'bid',
                    'from_curr': quote if side == 'bid' else base,
                    'to_curr': base if side == 'bid' else quote
                })
            else:
                print(f"  ✗ Not added to shortlist")
    
    print(f"\n{len(tested_detours)} detours tested, {len(shortlist)} viable detours shortlisted")
    
    if not shortlist:
        if direct_result['fill_rate'] < 100:
            print(f"\n⚠️  No viable detours found to improve {direct_result['fill_rate']:.1f}% fill rate")
            print("   Using direct route only (some volume will remain unfilled)")
        else:
            print("No cost-effective detours found, using direct route only")
        return {
            'routes': [pair],
            'allocation': [100],
            'total_cost': direct_total_cost,
            'avg_price': direct_result['avg_price'],
            'fill_rate': direct_result['fill_rate']
        }
    
    # Build route list
    routes = [
        {
            'name': f'{base}->{quote}' if side == 'ask' else f'{quote}->{base}',
            'segments': [{'pair': pair, 'side': side}],
            'stops': 0
        }
    ]
    
    # Sort shortlist by cost per unit (or revenue per unit for ask)
    if side == 'bid':
        shortlist.sort(key=lambda x: x['cost_per_unit'])
    else:
        shortlist.sort(key=lambda x: x['cost_per_unit'], reverse=True)
    
    for detour in shortlist[:3]:  # Top 3 detours
        if side == 'bid':
            route_name = f"{quote}->{detour['intermediate']}->{base}"
        else:
            route_name = f"{base}->{detour['intermediate']}->{quote}"
        
        routes.append({
            'name': route_name,
            'segments': detour['segments'],
            'stops': 1,
            'is_bid': detour.get('is_bid', False),
            'from_curr': detour.get('from_curr'),
            'to_curr': detour.get('to_curr')
        })
    
    # Step 4: Optimize allocation
    print(f"\n{'='*60}")
    print(f"STEP 3: Optimizing Allocation Across {len(routes)} Routes")
    print(f"{'='*60}")
    
    # Store direct route info for conversions
    direct_route_info = {
        'pair': pair,
        'side': side,
        'initial_microprice': direct_result['initial_microprice']
    }
    
    cache = {}
    
    # Phase 1: Test coarse grid
    grid_points = generate_smart_grid(len(routes))
    best_allocation = None
    best_cost = float('inf')
    best_details = None
    best_score = float('-inf')
    
    print(f"\nTesting {len(grid_points)} coarse grid points...")
    
    for i, allocation in enumerate(grid_points):
        cost, details = evaluate_route_combination(routes, allocation, volume, time_seconds, 
                                                  gemini_monthly_volume, cache, direct_route_info, verbose=True)
        
        # Score prioritizes fill rate heavily
        score = details['fill_rate'] * 1000 - cost
        
        if True: # i < 5 or score > best_score:  # Show first few and improvements
            alloc_str = ', '.join([f"{routes[j]['name']}: {int(allocation[j])}%" 
                                  for j in range(len(routes)) if allocation[j] > 0])
            
            # Calculate weighted average price
            total_weighted_price = 0
            total_filled = 0
            for route_detail in details['routes']:
                total_weighted_price += route_detail['avg_price'] * route_detail['filled']
                total_filled += route_detail['filled']
            
            avg_price_per_unit = total_weighted_price / total_filled if total_filled > 0 else float('inf')
            
            print(f"  [{alloc_str}] → Cost: {cost:.2f} ETH, Fill: {details['fill_rate']:.1f}%, Avg price: {avg_price_per_unit:.6f}")
        
        if score > best_score:
            best_score = score
            best_cost = cost
            best_allocation = allocation
            best_details = details
    
    print(f"\nBest allocation: {best_allocation} (fill: {best_details['fill_rate']:.1f}%)")
    
    # Phase 2: Coordinate descent refinement
    refinement_rounds = {'fast': 1, 'normal': 3, 'thorough': 5}[refinement_level]
    
    print(f"\nRefining with coordinate descent ({refinement_rounds} rounds)...")
    current_allocation = best_allocation.copy()
    
    for round_num in range(refinement_rounds):
        step = 20 // (round_num + 1)
        print(f"\nRound {round_num + 1} (step size: {step}%):")
        improved = True
        round_tests = 0
        
        while improved and step >= 5:
            improved = False
            
            for from_idx in range(len(routes)):
                for to_idx in range(len(routes)):
                    if from_idx == to_idx or current_allocation[from_idx] < step:
                        continue
                    
                    test = current_allocation.copy()
                    test[from_idx] -= step
                    test[to_idx] += step
                    
                    cost, details = evaluate_route_combination(routes, test, volume, time_seconds,
                                                             gemini_monthly_volume, cache, direct_route_info)
                    round_tests += 1
                    
                    # Score prioritizes fill rate
                    score = details['fill_rate'] * 1000 - cost
                    
                    if score > best_score:
                        print(f"  Improvement found: fill {best_details['fill_rate']:.1f}% → {details['fill_rate']:.1f}%")
                        current_allocation = test
                        best_score = score
                        best_cost = cost
                        best_details = details
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                step = step // 2
        
        print(f"  Tested {round_tests} moves this round")
    
    # Show final results
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")
    
    print(f"\nRoute Performance Summary:")
    print(f"{'='*60}")
    
    # First show the direct route at 100%
    direct_details = {
        'route': routes[0]['name'],
        'allocation': 100,
        'volume': volume,
        'filled': direct_result['executed_volume'],
        'avg_price': direct_result['avg_price'],
        'execution_cost': direct_result['total_cost'],
        'tx_fees': direct_tx_fee,
        'total_cost': direct_total_cost
    }
    
    print(f"\nDirect Route Only:")
    print(f"  Route: {direct_details['route']}")
    print(f"  Allocation: {direct_details['allocation']}%")
    print(f"  Volume: {direct_details['volume']:.2f} {base}")
    print(f"  Fill rate: {(direct_details['filled'] / direct_details['volume'] * 100):.1f}%")
    print(f"  Avg price per {base}: {direct_details['avg_price']:.6f} {quote}/{base}")
    print(f"  Execution cost: {direct_details['execution_cost']:.2f} {quote}")
    print(f"  Exchange fees: {direct_details['tx_fees']:.4f} {quote}")
    print(f"  Total cost: {direct_details['total_cost']:.2f} {quote}")
    
    # Now show the optimized allocation
    print(f"\nOptimized Multi-Route Execution:")
    print(f"{'='*60}")
    
    total_weighted_price = 0
    total_filled_volume = 0
    
    for route_detail in best_details['routes']:
        print(f"\nRoute: {route_detail['route']}")
        print(f"  Allocation: {route_detail['allocation']:.1f}%")
        print(f"  Volume: {route_detail['volume']:.2f} {base}")
        print(f"  Fill rate: {(route_detail['filled'] / route_detail['volume'] * 100 if route_detail['volume'] > 0 else 0):.1f}%")
        print(f"  Avg price per {base}: {route_detail['avg_price']:.6f} {quote}/{base}")
        print(f"  Execution cost: {route_detail['execution_cost']:.2f} {quote}")
        print(f"  Exchange fees: {route_detail['tx_fees']:.4f} {quote}")
        print(f"  Total cost: {route_detail['total_cost']:.2f} {quote}")
        
        # Accumulate for weighted average price
        total_weighted_price += route_detail['avg_price'] * route_detail['filled']
        total_filled_volume += route_detail['filled']
    
    # Calculate overall metrics
    overall_avg_price = total_weighted_price / total_filled_volume if total_filled_volume > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Overall Metrics:")
    print(f"  Total volume requested: {volume:.2f} {base}")
    print(f"  Total volume filled: {total_filled_volume:.2f} {base}")
    print(f"  Overall fill rate: {best_details['fill_rate']:.1f}%")
    print(f"  Weighted avg price: {overall_avg_price:.6f} {quote}/{base}")
    print(f"  Total execution cost: {best_details['execution_cost']:.2f} {quote}")
    print(f"  Total exchange fees: {best_details['tx_fees']:.4f} {quote}")
    print(f"  Total cost: {best_details['total_cost']:.2f} {quote}")
    
    # Recommendation
    print(f"\n{'='*60}")
    if best_details['fill_rate'] == 100 and direct_result['fill_rate'] == 100:
        if best_details['total_cost'] < direct_total_cost:
            savings = direct_total_cost - best_details['total_cost']
            print(f"✓ RECOMMENDATION: Use optimized routing")
            print(f"  Saves {savings:.4f} {quote} ({savings/direct_total_cost*100:.1f}%)")
            print(f"  Avg price improvement: {(direct_result['avg_price'] - overall_avg_price):.6f} {quote} per {base}")
        else:
            print(f"✓ RECOMMENDATION: Use direct route only")
            print(f"  Simpler and {(direct_total_cost - best_details['total_cost']):.4f} {quote} cheaper")
    elif best_details['fill_rate'] > direct_result['fill_rate']:
        print(f"✓ RECOMMENDATION: Use optimized routing")
        print(f"  Fills {(best_details['fill_rate'] - direct_result['fill_rate']):.1f}% more volume")
        extra_filled = total_filled_volume - direct_details['filled']
        print(f"  Additional {base} acquired: {extra_filled:.2f}")
        if total_filled_volume > 0:
            print(f"  Overall avg price: {overall_avg_price:.6f} {quote}/{base}")
    else:
        print(f"✓ RECOMMENDATION: Use direct route only")
        print(f"  Better fill rate and simpler execution")
    
    # Return structured results
    route_names = []
    final_allocation = []
    for i, (route, alloc) in enumerate(zip(routes, current_allocation)):
        if alloc > 0:
            route_names.append(route['name'])
            final_allocation.append(alloc)
    
    return {
        'routes': route_names,
        'allocation': final_allocation,
        'total_cost': best_cost,
        'fill_rate': best_details['fill_rate'],
        'evaluations': len(cache),
        'avg_price': overall_avg_price,
        'details': best_details
    }

if __name__ == "__main__":
    smart_order_route(
        pair='ETHBTC',
        side='ask',
        volume=50,
        time_seconds=12*60*60,
        refinement_level='thorough'
    )