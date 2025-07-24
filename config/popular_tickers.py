# Popular stock tickers for enhanced caching
# These tickers get longer cache TTL during market hours

POPULAR_TICKERS = [
    # FAANG + Top Tech
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA',
    
    # Major Tech
    'NFLX', 'AMD', 'INTC', 'ORCL', 'CRM', 'ADBE', 'PYPL', 'UBER', 'LYFT',
    
    # Financial
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP',
    
    # Healthcare & Pharma
    'JNJ', 'PFE', 'UNH', 'MRNA', 'ABBV', 'TMO', 'ABT', 'LLY',
    
    # Consumer & Retail  
    'WMT', 'TGT', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'DIS', 'AMGN',
    
    # Energy & Industrial
    'XOM', 'CVX', 'COP', 'BA', 'CAT', 'GE', 'MMM', 'HON',
    
    # Communication & Media
    'T', 'VZ', 'CMCSA', 'TMUS',
    
    # ETFs - Most Popular
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'IVV', 'VEA', 'IEFA', 'EEM',
    'GLD', 'SLV', 'TLT', 'HYG', 'LQD', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI',
    
    # Crypto-Related
    'COIN', 'MSTR', 'SQ', 'HOOD',
    
    # Electric Vehicle & Clean Energy  
    'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'ENPH', 'PLUG',
    
    # Meme Stocks & High Volume
    'GME', 'AMC', 'BB', 'NOK', 'PLTR', 'WISH', 'CLOV',
    
    # Biotech
    'GILD', 'BIIB', 'REGN', 'VRTX', 'ILMN',
    
    # Other High Volume
    'F', 'GEM', 'SNAP', 'TWTR', 'PINS', 'ZM', 'ROKU', 'PTON', 'SHOP',
    'ARKK', 'ARKQ', 'ARKG', 'ARKW', 'SQQQ', 'TQQQ', 'UVXY', 'VIX'
]

# Categorized for potential future use
CATEGORIES = {
    'mega_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA'],
    'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'AMD', 'INTC'],
    'finance': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA'],
    'etfs': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO'],
    'meme': ['GME', 'AMC', 'BB', 'NOK', 'PLTR'],
    'ev': ['TSLA', 'NIO', 'XPEV', 'LI', 'RIVN', 'LCID']
}

def is_popular_ticker(ticker: str) -> bool:
    """Check if a ticker is in the popular list."""
    return ticker.upper() in POPULAR_TICKERS

def get_ticker_category(ticker: str) -> str:
    """Get the category of a ticker."""
    ticker = ticker.upper()
    for category, tickers in CATEGORIES.items():
        if ticker in tickers:
            return category
    return 'other'
