#!/usr/bin/env python3
"""
Smart Order Router - Fixed Version with Correct Time Allocation
Each route gets full time duration, allocation only affects volume
"""

from twap_estimator import calculate_execution_cost, split_pair
from symbols import SYMBOLS_SET, is_valid_pair
import numpy as np
from typing import Dict, List, Tuple, Optional

def check_pair_exists(base: str, quote: str) -> Optional[str]:
    """
    Check if a trading pair exists and return the valid pair format.
    Returns the pair string if exists, None otherwise.
    Uses O(1) lookup instead of calling get_market_data.
    """
    # Try both orderings
    pair1 = base + quote
    pair2 = quote + base
    
    if is_valid_pair(pair1):
        return pair1
    elif is_valid_pair(pair2):
        return pair2
    return None

def find_detour_routes(from_curr: str, to_curr: str) -> List[Dict]:
    """
    Find all valid 2-leg detour routes for trading from_curr → to_curr.
    Now using O(1) lookups against the valid pairs set.
    """
    detours = []
    
    for intermediate in SYMBOLS_SET:
        if intermediate in [from_curr, to_curr]:
            continue
        
        # Check if we can trade between to_curr and intermediate
        leg1_pair = check_pair_exists(to_curr, intermediate)
        if not leg1_pair:
            continue
            
        # Check if we can trade between intermediate and from_curr
        leg2_pair = check_pair_exists(intermediate, from_curr)
        if not leg2_pair:
            continue
        
        detours.append({
            'intermediate': intermediate,
            'leg1_pair': leg1_pair,
            'leg2_pair': leg2_pair
        })
    
    return detours

def determine_leg_side(pair: str, from_curr: str, to_curr: str) -> str:
    """
    Determine if we're buying or selling in a given pair.
    """
    pair_base, pair_quote = split_pair(pair)
    
    if from_curr == pair_base and to_curr == pair_quote:
        return 'sell'
    elif from_curr == pair_quote and to_curr == pair_base:
        return 'buy'
    else:
        raise ValueError(f"Cannot determine side for {pair} going from {from_curr} to {to_curr}")

def execute_route(route: Dict, volume: float, time_seconds: int, use_live: bool = False) -> Dict:
    """
    Execute a trade on a specific route (direct or detour).
    Returns dict with total_cost, executed_volume, and fill_rate.
    
    IMPORTANT: For detour routes, each leg gets half the total time since they execute sequentially.
    """
    if route['type'] == 'direct':
        result = calculate_execution_cost(
            route['pair'],
            route['side'], 
            volume,
            time_seconds,  # Direct route gets full time
            use_live=use_live
        )
        
        if 'error' in result:
            return {'error': result['error'], 'fill_rate': 0, 'total_cost': float('inf'), 'executed_volume': 0}
        
        return {
            'total_cost': result['total_cost'],
            'executed_volume': result['executed_volume'],
            'fill_rate': result['fill_rate'],
            'avg_price': result['avg_price']
        }
    
    elif route['type'] == 'detour':
        # For detour routes, split time between two legs (sequential execution)
        leg_time = time_seconds // 2
        
        if route['original_side'] == 'buy':
            # BUY: Work backwards from target
            leg1_result = calculate_execution_cost(
                route['leg1_pair'],
                route['leg1_side'],
                volume,
                leg_time,  # Each leg gets half the time
                use_live=use_live
            )
            
            if 'error' in leg1_result:
                return {'error': 'Leg 1 error', 'fill_rate': 0, 'total_cost': float('inf'), 'executed_volume': 0}
            
            intermediate_needed = leg1_result['total_cost']
            
            leg2_result = calculate_execution_cost(
                route['leg2_pair'],
                route['leg2_side'],
                intermediate_needed,
                leg_time,  # Each leg gets half the time
                use_live=use_live
            )
            
            if 'error' in leg2_result:
                return {'error': 'Leg 2 error', 'fill_rate': 0, 'total_cost': float('inf'), 'executed_volume': 0}
            
            return {
                'total_cost': leg2_result['total_cost'],
                'executed_volume': leg1_result['executed_volume'],
                'fill_rate': min(leg1_result['fill_rate'], leg2_result['fill_rate']),
                'avg_price': leg2_result['total_cost'] / leg1_result['executed_volume'] if leg1_result['executed_volume'] > 0 else float('inf')
            }
        
        else:  # SELL
            leg1_result = calculate_execution_cost(
                route['leg2_pair'],
                route['leg1_side'],
                volume,
                leg_time,  # Each leg gets half the time
                use_live=use_live
            )
            
            if 'error' in leg1_result:
                return {'error': 'Leg 1 error', 'fill_rate': 0, 'total_cost': float('inf'), 'executed_volume': 0}
            
            if route['leg1_side'] == 'sell':
                intermediate_received = leg1_result['total_cost'] / leg1_result['avg_price']
            else:
                intermediate_received = leg1_result['executed_volume']
            
            leg2_result = calculate_execution_cost(
                route['leg1_pair'],
                route['leg2_side'],
                intermediate_received,
                leg_time,  # Each leg gets half the time
                use_live=use_live
            )
            
            if 'error' in leg2_result:
                return {'error': 'Leg 2 error', 'fill_rate': 0, 'total_cost': float('inf'), 'executed_volume': 0}
            
            if route['leg2_side'] == 'sell':
                total_received = leg2_result['total_cost'] / leg2_result['avg_price']
            else:
                total_received = leg2_result['executed_volume']
            
            return {
                'total_cost': volume,
                'executed_volume': total_received,
                'fill_rate': min(leg1_result['fill_rate'], leg2_result['fill_rate']),
                'avg_price': total_received / volume if volume > 0 else 0
            }
    
    return {'error': 'Unknown route type', 'fill_rate': 0, 'total_cost': float('inf'), 'executed_volume': 0}

def optimize_allocation_greedy(routes: List[Dict], volume: float, time_seconds: int, 
                               use_live: bool = False) -> Dict:
    """
    Optimize allocation using greedy approach.
    Start with best single route, then add others to improve.
    Priority 1: Maximize fill rate
    Priority 2: Minimize unit cost
    
    FIXED: Each route gets full time_seconds, not proportional to allocation.
    FIXED: Proper route selection and transfer logic.
    """
    n_routes = len(routes)
    
    if n_routes == 1:
        return {
            'allocation': [100.0],
            'routes': routes,
            'details': None
        }
    
    def evaluate_allocation(alloc: List[float]) -> Tuple[float, float, float, Dict]:
        """
        Evaluate allocation and return (fill_rate, unit_cost, total_cost, details)
        FIXED: Each route gets full time, allocation only affects volume
        """
        assert abs(sum(alloc) - 100.0) < 0.01, f"Allocation sums to {sum(alloc)}, not 100%"
        
        total_cost = 0
        total_filled = 0
        route_results = []
        
        for i, route in enumerate(routes):
            if alloc[i] < 0.1:  # Skip routes with negligible allocation
                continue
                
            route_volume = volume * alloc[i] / 100
            # FIX: Each route gets FULL time, not proportional to allocation
            # This reflects parallel execution across routes
            route_time = time_seconds  # ✅ FIXED: Full time for each route
            
            result = execute_route(route, route_volume, route_time, use_live)
            
            if 'error' not in result:
                total_cost += result['total_cost']
                total_filled += result['executed_volume']
                route_results.append({
                    'route': route['name'],
                    'allocation': alloc[i],
                    'volume': route_volume,
                    'filled': result['executed_volume'],
                    'cost': result['total_cost'],
                    'fill_rate': result['fill_rate']
                })
        
        fill_rate = (total_filled / volume) * 100 if volume > 0 else 0
        unit_cost = total_cost / total_filled if total_filled > 0 else float('inf')
        
        return fill_rate, unit_cost, total_cost, {
            'total_cost': total_cost,
            'total_filled': total_filled,
            'fill_rate': fill_rate,
            'unit_cost': unit_cost,
            'avg_price': unit_cost,
            'routes': route_results
        }
    
    # Step 1: Test each route at 100% to find the best starting point
    print("\n--- Testing each route at 100% ---")
    
    # Evaluate all routes and store results WITH route names for debugging
    route_results = []
    for i, route in enumerate(routes):
        alloc = [0.0] * n_routes
        alloc[i] = 100.0
        
        fill_rate, unit_cost, total_cost, details = evaluate_allocation(alloc)
        print(f"[{i}] {route['name']}: Fill {fill_rate:.1f}%, Unit cost {unit_cost:.6f}")
        
        route_results.append({
            'index': i,
            'name': route['name'],  # Store name for debugging
            'fill_rate': fill_rate,
            'unit_cost': unit_cost,
            'total_cost': total_cost,
            'details': details
        })
    
    # Debug: Show what we're comparing
    print("\n--- Debug: Route selection process ---")
    for r in route_results:
        print(f"  Index {r['index']} ({r['name']}): Fill={r['fill_rate']:.1f}%, Cost={r['unit_cost']:.6f}")
    
    # FIX: Since all routes typically have 100% fill rate, use simple selection by unit cost
    # Find the route with minimum unit cost
    best_route_idx = 0
    best_unit_cost = route_results[0]['unit_cost']
    best_fill_rate = route_results[0]['fill_rate']
    
    for i, result in enumerate(route_results):
        # Priority 1: Better fill rate
        # Priority 2: Same fill but better unit cost
        if result['fill_rate'] > best_fill_rate:
            best_route_idx = i
            best_unit_cost = result['unit_cost']
            best_fill_rate = result['fill_rate']
        elif abs(result['fill_rate'] - best_fill_rate) < 0.001 and result['unit_cost'] < best_unit_cost:
            best_route_idx = i
            best_unit_cost = result['unit_cost']
    
    print(f"Selected route index {best_route_idx}: {routes[best_route_idx]['name']}")
    
    # Set initial allocation
    allocation = [0.0] * n_routes
    allocation[best_route_idx] = 100.0
    
    # Get current state from fresh evaluation (to ensure consistency)
    best_fill_rate, best_unit_cost, best_total_cost, best_details = evaluate_allocation(allocation)
    
    print(f"\nStarting with {routes[best_route_idx]['name']} at 100%")
    print(f"  Fill rate: {best_fill_rate:.1f}%, Unit cost: {best_unit_cost:.6f}")
    
    # Step 2: Greedy improvement with decreasing step sizes
    # FIX 3: Remove 10% step for efficiency
    step_sizes = [20, 5]
    
    for step in step_sizes:
        print(f"\n--- Trying to add {step}% from other routes ---")
        improved = True
        iterations = 0
        
        while improved and iterations < 10:
            improved = False
            iterations += 1
            
            # Find the route with allocation to give from
            donor_idx = -1
            for i in range(n_routes):
                if allocation[i] >= step:
                    donor_idx = i
                    break
            
            if donor_idx == -1:
                break
            
            print(f"  Iteration {iterations}: Testing transfers from {routes[donor_idx]['name']}")
            
            # Collect ALL transfer options and compare them
            transfer_options = []
            
            for receiver_idx in range(n_routes):
                if receiver_idx == donor_idx:
                    continue
                
                # Try the transfer
                test_alloc = allocation.copy()
                test_alloc[donor_idx] -= step
                test_alloc[receiver_idx] += step
                
                fill_rate, unit_cost, total_cost, details = evaluate_allocation(test_alloc)
                
                transfer_options.append({
                    'donor_idx': donor_idx,
                    'receiver_idx': receiver_idx,
                    'allocation': test_alloc.copy(),
                    'fill_rate': fill_rate,
                    'unit_cost': unit_cost,
                    'total_cost': total_cost,
                    'details': details
                })
                
                print(f"    Option: Transfer to {routes[receiver_idx]['name']}: Fill {fill_rate:.1f}%, Cost {unit_cost:.6f}")
            
            # Find the best transfer option that actually improves things
            best_transfer = None
            best_new_fill = best_fill_rate
            best_new_unit_cost = best_unit_cost
            best_new_details = None
            
            for option in transfer_options:
                # Check if this option improves over current state
                improves = False
                if option['fill_rate'] > best_fill_rate + 0.001:  # Better fill
                    improves = True
                elif abs(option['fill_rate'] - best_fill_rate) <= 0.001 and option['unit_cost'] < best_unit_cost - 0.000001:
                    improves = True
                
                if improves:
                    # Check if this is the best improvement so far
                    if (option['fill_rate'] > best_new_fill + 0.001) or \
                       (abs(option['fill_rate'] - best_new_fill) <= 0.001 and option['unit_cost'] < best_new_unit_cost):
                        best_transfer = option
                        best_new_fill = option['fill_rate']
                        best_new_unit_cost = option['unit_cost']
                        best_new_details = option['details']
            
            # Apply best transfer if found
            if best_transfer is not None:
                donor_name = routes[best_transfer['donor_idx']]['name']
                receiver_name = routes[best_transfer['receiver_idx']]['name']
                print(f"  ✓ Selected: Transfer {step}% from {donor_name} to {receiver_name}")
                print(f"    New allocation: {[f'{a:.1f}%' for a in best_transfer['allocation']]}")
                print(f"    Fill: {best_new_fill:.1f}% (was {best_fill_rate:.1f}%)")
                print(f"    Unit cost: {best_new_unit_cost:.6f} (was {best_unit_cost:.6f})")
                
                # Update state
                allocation = best_transfer['allocation']
                best_fill_rate = best_new_fill
                best_unit_cost = best_new_unit_cost
                best_details = best_new_details
                improved = True
            else:
                print(f"    No improving transfers found")
        
        if not improved:
            print(f"  No improvements found with {step}% transfers")
    
    # Final summary
    final_alloc_strs = []
    for r, a in zip(routes, allocation):
        if a > 0:
            final_alloc_strs.append(f"{r['name']}: {a:.1f}%")
    
    print(f"\nFinal allocation: {', '.join(final_alloc_strs)}")
    print(f"Final fill rate: {best_fill_rate:.1f}%")
    print(f"Final unit cost: {best_unit_cost:.6f}")
    
    # Sanity check: Ensure we didn't make things worse
    if len(final_alloc_strs) > 1:
        # Check if splitting actually improved vs single best route
        single_best = min(route_results, key=lambda x: x['unit_cost'])
        if best_unit_cost > single_best['unit_cost'] + 0.000001 and abs(best_fill_rate - single_best['fill_rate']) < 0.001:
            print(f"\n⚠️ WARNING: Optimization made things worse!")
            print(f"  Best single route ({routes[single_best['index']]['name']}): {single_best['unit_cost']:.6f}")
            print(f"  Current allocation: {best_unit_cost:.6f}")
            print(f"  Difference: +{(best_unit_cost - single_best['unit_cost']):.6f}")
    
    return {
        'allocation': allocation,
        'routes': routes,
        'details': best_details
    }

def smart_order_route(pair: str, side: str, volume: float, time_seconds: int,
                     use_live: bool = False, optimize: bool = True) -> Dict:
    """
    Smart order routing algorithm with greedy allocation optimization.
    """
    print(f"\n{'='*60}")
    print(f"Smart Order Routing: {pair} {side.upper()} {volume} over {time_seconds}s")
    print(f"{'='*60}")
    
    # Step 1: Validate and split the input pair
    base, quote = split_pair(pair)
    if not base or not quote:
        return {'error': f'Invalid pair: {pair}'}
    
    print(f"Parsed pair: {base}/{quote}")
    
    # Determine trading direction
    if side.lower() == 'buy':
        from_curr = quote
        to_curr = base
    else:
        from_curr = base
        to_curr = quote
    
    print(f"Trading direction: {from_curr} → {to_curr}")
    
    all_routes = []
    
    # Step 2: Check direct route
    print(f"\n--- Checking Direct Route ---")
    
    direct_pair = check_pair_exists(base, quote)
    if direct_pair:
        direct_side = determine_leg_side(direct_pair, from_curr, to_curr)
        all_routes.append({
            'name': 'Direct',
            'type': 'direct',
            'pair': direct_pair,
            'side': direct_side,
            'original_side': side.lower()
        })
        print(f"Added direct route: {direct_pair} {direct_side}")
    else:
        print(f"No direct route available")
    
    # Step 3: Find detour routes
    print(f"\n--- Finding Detour Routes ---")
    detours = find_detour_routes(from_curr, to_curr)
    
    print(f"Found {len(detours)} possible detours")
    
    # Add detour routes (limit to top 5 for performance)
    for detour in detours[:5]:
        intermediate = detour['intermediate']
        
        # Determine leg sides
        if side.lower() == 'buy':
            leg1_side = determine_leg_side(detour['leg1_pair'], intermediate, to_curr)
            leg2_side = determine_leg_side(detour['leg2_pair'], from_curr, intermediate)
        else:
            leg1_side = determine_leg_side(detour['leg2_pair'], from_curr, intermediate)
            leg2_side = determine_leg_side(detour['leg1_pair'], intermediate, to_curr)
        
        all_routes.append({
            'name': f'via {intermediate}',
            'type': 'detour',
            'intermediate': intermediate,
            'leg1_pair': detour['leg1_pair'],
            'leg2_pair': detour['leg2_pair'],
            'leg1_side': leg1_side,
            'leg2_side': leg2_side,
            'original_side': side.lower()
        })
        print(f"Added detour: {from_curr} → {intermediate} → {to_curr}")
    
    # Step 4: Optimize or return error
    if not all_routes:
        return {'error': 'No valid routes found'}
    
    if len(all_routes) == 1:
        route_name = all_routes[0]['name']
        print(f"\n✓ Only one route available: {route_name}")
        result = execute_route(all_routes[0], volume, time_seconds, use_live)
        return {
            'route': route_name,
            'allocation': [100],
            'details': result
        }
    
    if optimize:
        print(f"\n--- Optimizing Allocation (Greedy) ---")
        print(f"Priority 1: Maximize fill rate")
        print(f"Priority 2: Minimize unit cost")
        
        result = optimize_allocation_greedy(all_routes, volume, time_seconds, use_live)
        
        print(f"\n✓ Optimization complete")
        return result
    else:
        # Find best single route
        print(f"\n--- Finding Best Single Route ---")
        best_route = None
        best_fill = 0
        best_unit_cost = float('inf')
        
        for route in all_routes:
            result = execute_route(route, volume, time_seconds, use_live)
            if 'error' not in result:
                fill = result['fill_rate']
                unit_cost = result['total_cost'] / result['executed_volume'] if result['executed_volume'] > 0 else float('inf')
                
                if (fill > best_fill) or (fill == best_fill and unit_cost < best_unit_cost):
                    best_route = route
                    best_fill = fill
                    best_unit_cost = unit_cost
                    best_result = result
        
        if best_route:
            print(f"\n✓ Best single route: {best_route['name']}")
            return {
                'route': best_route['name'],
                'allocation': [100],
                'details': best_result
            }
        else:
            return {'error': 'No viable routes found'}

if __name__ == "__main__":
    result = smart_order_route(
        pair='LINKBTC',
        side='sell',
        volume=100,
        time_seconds=3600,
        use_live=False,
        optimize=True
    )
    
    if 'error' not in result:
        print(f"\n{'='*60}")
        print("FINAL RESULT")
        print(f"{'='*60}")
        if 'routes' in result:
            for route, alloc in zip(result['routes'], result['allocation']):
                if alloc > 0:
                    route_name = route['name']
                    print(f"{route_name}: {alloc:.1f}%")
        print(f"\nOverall metrics:")
        if result.get('details'):
            details = result['details']
            print(f"  Fill rate: {details['fill_rate']:.1f}%")
            print(f"  Unit cost: {details.get('unit_cost', details.get('avg_price', 0)):.6f}")
            print(f"  Total cost: {details['total_cost']:.6f}")