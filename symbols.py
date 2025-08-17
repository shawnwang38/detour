"""
Valid currency symbols and trading pairs supported by Gemini
"""
from typing import Optional

# Individual currency symbols
SYMBOLS = [
    # Major currencies
    'BTC', 'ETH', 'USD', 'EUR', 'GBP', 'SGD',
    
    # Stablecoins
    'USDT', 'USDC', 'GUSD', 'DAI', 'PAX', 'RLUSD',
    
    # Altcoins (3-letter)
    'SOL', 'DOT', 'UNI', 'LTC', 'BCH', 'XRP', 'ZEC', 'BAT', 'MKR',
    'CRV', 'YFI', 'AMP', 'FIL', 'GRT', 'LRC', 'QNT', 'IMX', 'APE',
    'INJ', 'FTM', 'GMT', 'HNT', 'LDO', 'LPT', 'OP', 'API', 'ARB',
    'CHZ', 'CTX', 'ENS', 'EUL', 'FET', 'JTO', 'MEW', 'SKL',
    'UMA', 'WIF', 'XTZ', 'ALI', 'WCT',
    
    # Altcoins (4-letter)
    'LINK', 'AAVE', 'AVAX', 'DOGE', 'ATOM', 'MATIC', 'SHIB', 'MANA',
    'SAND', 'ANKR', 'COMP', 'MASK', 'SUSHI', 'STORJ', 'IOTX', 'GALA',
    'PAXG', 'RNDR', 'CUBE', 'SAMO', 'BONK', 'PEPE', 'FLOKI', 'BOME',
    'ELON', 'GOAT', 'PNUT', 'PUMP', 'PYTH', 'PENGU',
    
    # Longer symbols
    'POPCAT', 'TRUMP', 'JITOSOL', 'MOODENG', 'CHILLGUY'
]

# Create a set for O(1) lookup
SYMBOLS_SET = set(SYMBOLS)

# Dictionary mapping REVERSED symbols to their GEMINI API symbols
# This is what get_market_data uses internally
REVERSED_TO_GEMINI = {
    "gusdaave": "aavegusd",
    "usdaave": "aaveusd",
    "gusdali": "aligusd",
    "usdali": "aliusd",
    "gusdamp": "ampgusd",
    "usdamp": "ampusd",
    "gusdankr": "ankrgusd",
    "usdankr": "ankrusd",
    "gusdape": "apegusd",
    "usdape": "apeusd",
    "gusdapi3": "api3gusd",
    "usdapi3": "api3usd",
    "gusdarb": "arbgusd",
    "usdarb": "arbusd",
    "gusdatom": "atomgusd",
    "usdatom": "atomusd",
    "gusdavax": "avaxgusd",
    "usdavax": "avaxusd",
    "gusdbat": "batgusd",
    "usdbat": "batusd",
    "gusdbch": "bchgusd",
    "usdbch": "bchusd",
    "gusdbome": "bomegusd",
    "usdbome": "bomeusd",
    "gusdbonk": "bonkgusd",
    "usdbonk": "bonkusd",
    "eurbtc": "btceur",
    "gbpbtc": "btcgbp",
    "gusdbtc": "btcgusd",
    "sgdbtc": "btcsgd",
    "usdbtc": "btcusd",
    "usdtbtc": "btcusdt",
    "gusdchillguy": "chillguygusd",
    "usdchillguy": "chillguyusd",
    "gusdchz": "chzgusd",
    "usdchz": "chzusd",
    "gusdcomp": "compgusd",
    "usdcomp": "compusd",
    "gusdcrv": "crvgusd",
    "usdcrv": "crvusd",
    "gusdctx": "ctxgusd",
    "usdctx": "ctxusd",
    "gusdcube": "cubegusd",
    "usdcube": "cubeusd",
    "gusddai": "daigusd",
    "usddai": "daiusd",
    "btcdoge": "dogebtc",
    "ethdoge": "dogeeth",
    "gusddoge": "dogegusd",
    "usddoge": "dogeusd",
    "gusddot": "dotgusd",
    "usddot": "dotusd",
    "gusdelon": "elongusd",
    "usdelon": "elonusd",
    "gusdens": "ensgusd",
    "usdens": "ensusd",
    "btceth": "ethbtc",
    "eureth": "etheur",
    "gbpeth": "ethgbp",
    "gusdeth": "ethgusd",
    "sgdeth": "ethsgd",
    "usdeth": "ethusd",
    "usdteth": "ethusdt",
    "usdeul": "eulusd",
    "gusdfet": "fetgusd",
    "usdfet": "fetusd",
    "gusdfil": "filgusd",
    "usdfil": "filusd",
    "gusdfloki": "flokigusd",
    "usdfloki": "flokiusd",
    "gusdftm": "ftmgusd",
    "usdftm": "ftmusd",
    "gusdgala": "galagusd",
    "usdgala": "galausd",
    "gusdgmt": "gmtgusd",
    "usdgmt": "gmtusd",
    "gusdgoat": "goatgusd",
    "usdgoat": "goatusd",
    "gusdgrt": "grtgusd",
    "usdgrt": "grtusd",
    "gbpgusd": "gusdgbp",
    "sgdgusd": "gusdsgd",
    "gusdhnt": "hntgusd",
    "usdhnt": "hntusd",
    "gusdimx": "imxgusd",
    "usdimx": "imxusd",
    "gusdinj": "injgusd",
    "usdinj": "injusd",
    "gusdiotx": "iotxgusd",
    "usdiotx": "iotxusd",
    "soljitosol": "jitosolsol",
    "usdjitosol": "jitosolusd",
    "usdjto": "jtousd",
    "gusdldo": "ldogusd",
    "usdldo": "ldousd",
    "btclink": "linkbtc",
    "ethlink": "linketh",
    "gusdlink": "linkgusd",
    "usdlink": "linkusd",
    "gusdlpt": "lptgusd",
    "usdlpt": "lptusd",
    "gusdlrc": "lrcgusd",
    "usdlrc": "lrcusd",
    "btcltc": "ltcbtc",
    "ethltc": "ltceth",
    "gusdltc": "ltcgusd",
    "usdltc": "ltcusd",
    "gusdmana": "managusd",
    "usdmana": "manausd",
    "gusdmask": "maskgusd",
    "usdmask": "maskusd",
    "gusdmatic": "maticgusd",
    "usdmatic": "maticusd",
    "gusdmew": "mewgusd",
    "usdmew": "mewusd",
    "gusdmkr": "mkrgusd",
    "usdmkr": "mkrusd",
    "gusdmoodeng": "moodenggusd",
    "usdmoodeng": "moodengusd",
    "gusdop": "opgusd",
    "usdop": "opusd",
    "gusdpaxg": "paxggusd",
    "gusdpax": "paxgusd",
    "usdpengu": "penguusd",
    "gusdpepe": "pepegusd",
    "usdpepe": "pepeusd",
    "gusdpnut": "pnutgusd",
    "usdpnut": "pnutusd",
    "gusdpopcat": "popcatgusd",
    "usdpopcat": "popcatusd",
    "usdpump": "pumpusd",
    "gusdpyth": "pythgusd",
    "usdpyth": "pythusd",
    "gusdqnt": "qntgusd",
    "usdqnt": "qntusd",
    "usdrlusd": "rlusdusd",
    "gusdrndr": "rndrgusd",
    "usdrndr": "rndrusd",
    "gusdsamo": "samogusd",
    "usdsamo": "samousd",
    "gusdsand": "sandgusd",
    "usdsand": "sandusd",
    "gusdshib": "shibgusd",
    "usdshib": "shibusd",
    "gusdskl": "sklgusd",
    "usdskl": "sklusd",
    "btcsol": "solbtc",
    "ethsol": "soleth",
    "gusdsol": "solgusd",
    "usdsol": "solusd",
    "gusdstorj": "storjgusd",
    "usdstorj": "storjusd",
    "gusdsushi": "sushigusd",
    "usdsushi": "sushiusd",
    "gusdtrump": "trumpgusd",
    "usdtrump": "trumpusd",
    "gusduma": "umagusd",
    "usduma": "umausd",
    "gusduni": "unigusd",
    "usduni": "uniusd",
    "usdusdc": "usdcusd",
    "gusdusdt": "usdtgusd",
    "usdusdt": "usdtusd",
    "usdwct": "wctusd",
    "gusdwif": "wifgusd",
    "usdwif": "wifusd",
    "gusdxrp": "xrpgusd",
    "rlusddxrp": "xrprlusd",
    "usdxrp": "xrpusd",
    "gusdxtz": "xtzgusd",
    "usdxtz": "xtzusd",
    "gusdyfi": "yfigusd",
    "usdyfi": "yfiusd",
    "gusdzec": "zecgusd",
    "usdzec": "zecusd"
}

# Create a set of all valid Gemini pairs (both forms) for O(1) lookup
VALID_PAIRS = set(REVERSED_TO_GEMINI.keys()) | set(REVERSED_TO_GEMINI.values())

def is_valid_pair(pair: str) -> bool:
    """Check if a pair is valid on Gemini (O(1) lookup)"""
    return pair.lower() in VALID_PAIRS

def get_gemini_symbol(pair: str) -> Optional[str]:
    """Get the Gemini API symbol for a pair, handling reversed notation"""
    pair_lower = pair.lower()
    
    # Direct match with Gemini format
    if pair_lower in REVERSED_TO_GEMINI.values():
        return pair_lower
    
    # Reversed notation that needs mapping
    if pair_lower in REVERSED_TO_GEMINI:
        return REVERSED_TO_GEMINI[pair_lower]
    
    return None