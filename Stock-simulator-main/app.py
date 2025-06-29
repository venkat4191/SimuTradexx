import os
import secrets
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from dotenv import load_dotenv
import json
import re
from ai_predictor import StockAI
from werkzeug.security import check_password_hash, generate_password_hash

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'postgresql://localhost/stock_simulator')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    balance = db.Column(db.Float, default=100000.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    portfolios = db.relationship('Portfolio', backref='user', lazy=True)

class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    shares = db.Column(db.Integer, nullable=False)
    avg_price = db.Column(db.Float, nullable=False)
    purchased_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create database tables
with app.app_context():
    db.create_all()

# The get_news function fetches the latest business news from News API
# The display_portfolio function generates a summary of the user's portfolio including current values and profit/loss
# The generate_portfolio_pie_chart function creates a pie chart to visualize portfolio performance
# The buy_stock and sell_stock functions handle buying and selling of stocks respectively

def get_news():
    """Get news from RSS feeds instead of NewsAPI"""
    news_sources = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^NSEI&region=IN&lang=en-IN",
        "https://www.moneycontrol.com/rss/business.xml",
        "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
        "https://www.livemint.com/rss/markets"
    ]
    
    all_articles = []
    
    for source in news_sources:
        try:
            print(f"Fetching news from: {source}")
            feed = feedparser.parse(source)
            
            for entry in feed.entries[:3]:  # Get top 3 from each source
                article = {
                    "title": entry.get('title', 'No title'),
                    "description": entry.get('summary', 'No description available'),
                    "url": entry.get('link', '#'),
                    "publishedAt": entry.get('published', datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'))
                }
                all_articles.append(article)
                
        except Exception as e:
            print(f"Error fetching from {source}: {e}")
            continue
    
    # If no RSS feeds work, return sample data
    if not all_articles:
        print("All RSS feeds failed, using sample data")
        return [
            {
                "title": "Stock Market Shows Strong Performance",
                "description": "Indian markets continue to show resilience with major indices reaching new highs. Analysts predict continued growth in the coming quarters.",
                "url": "https://www.moneycontrol.com/news/business/markets/",
                "publishedAt": "2025-06-28T10:00:00Z"
            },
            {
                "title": "Tech Sector Leads Market Gains",
                "description": "Technology stocks are driving market momentum with strong quarterly earnings reports from major IT companies.",
                "url": "https://economictimes.indiatimes.com/markets/stocks/news",
                "publishedAt": "2025-06-28T09:30:00Z"
            },
            {
                "title": "Banking Sector Sees Positive Trends",
                "description": "Banking stocks are performing well with improved asset quality and strong credit growth numbers.",
                "url": "https://www.livemint.com/market",
                "publishedAt": "2025-06-28T09:00:00Z"
            },
            {
                "title": "Market Volatility Expected This Week",
                "description": "Traders should brace for increased volatility as key economic data is set to be released this week.",
                "url": "https://www.moneycontrol.com/news/business/markets/",
                "publishedAt": "2025-06-28T08:30:00Z"
            },
            {
                "title": "Investment Opportunities in Emerging Sectors",
                "description": "Analysts identify promising investment opportunities in renewable energy and electric vehicle sectors.",
                "url": "https://economictimes.indiatimes.com/markets/stocks/news",
                "publishedAt": "2025-06-28T08:00:00Z"
            }
        ]
    
    print(f"Successfully fetched {len(all_articles)} articles from RSS feeds")
    return all_articles[:10]  # Return max 10 articles

def display_portfolio(portfolio_dict, user_balance):
    portfolio_info = {
        'total_value': 0,
        'total_investment': 0,
        'return_percentage': 0,
        'holdings': [],
        'best_performer': {'symbol': 'N/A', 'return_percentage': 0},
        'worst_performer': {'symbol': 'N/A', 'return_percentage': 0},
        'beta': 0,
        'sharpe_ratio': 0
    }
    
    total_investment = 0
    current_portfolio_value = 0
    best_return = float('-inf')
    worst_return = float('inf')
    
    for symbol, shares in portfolio_dict.items():
        try:
            stock = yf.Ticker(symbol + ".NS")
            current_price = stock.history(period="1d")["Close"].iloc[-1]
            current_value = current_price * shares
            
            # Get average purchase price from database
            user_portfolio_items = Portfolio.query.filter_by(user_id=current_user.id, symbol=symbol).all()
            total_purchased_value = sum(item.avg_price * item.shares for item in user_portfolio_items)
            total_purchased_shares = sum(item.shares for item in user_portfolio_items)
            avg_purchase_price = total_purchased_value / total_purchased_shares if total_purchased_shares > 0 else 0
            
            purchased_value = avg_purchase_price * shares
            total_investment += purchased_value
            current_portfolio_value += current_value
            
            return_percentage = ((current_value - purchased_value) / purchased_value) * 100 if purchased_value > 0 else 0
            
            # Update best and worst performers
            if return_percentage > best_return:
                best_return = return_percentage
                portfolio_info['best_performer'] = {'symbol': symbol, 'return_percentage': return_percentage}
            if return_percentage < worst_return:
                worst_return = return_percentage
                portfolio_info['worst_performer'] = {'symbol': symbol, 'return_percentage': return_percentage}
            
            # Add holding information
            portfolio_info['holdings'].append({
                'symbol': symbol,
                'quantity': shares,
                'avg_price': round(avg_purchase_price, 2),
                'current_price': round(current_price, 2),
                'value': round(current_value, 2),
                'return_percentage': round(return_percentage, 2),
                'percentage': round((current_value / current_portfolio_value * 100), 2) if current_portfolio_value > 0 else 0
            })
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    
    # Calculate portfolio metrics
    portfolio_info['total_value'] = round(current_portfolio_value, 2)
    portfolio_info['total_investment'] = round(total_investment, 2)
    portfolio_info['return_percentage'] = round(((current_portfolio_value - total_investment) / total_investment * 100), 2) if total_investment > 0 else 0
    
    # Generate pie chart for portfolio performance
    generate_portfolio_pie_chart(portfolio_dict)
    
    return portfolio_info

def generate_portfolio_pie_chart(portfolio):
    try:
        labels = list(portfolio.keys())
        sizes = list(portfolio.values())

        plt.figure(figsize=(6, 6))  # It indicates the figure size
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')  # Equal means it ensures the pie chart is circle 
        plt.title('Portfolio Performance')
        plt.savefig('static/portfolio_pie_chart.png')  # Save the pie chart as a PNG file
        plt.close()
    except Exception as e:
        print(f"Error generating pie chart: {e}")

def buy_stock(portfolio, symbol, shares):
    global balance
    symbol = symbol.upper()  # Always use uppercase, no .NS in portfolio
    print("[BUY] Before:", portfolio)
    try:
        stock = yf.Ticker(symbol + ".NS")
        price = stock.history(period="1d")["Close"].iloc[-1]
        if symbol in venkat:
            venkat[symbol] = float(f"{price:.2f}")
        else:
            venkat[symbol] = float(f"{price:.2f}")

        total_cost = price * shares
        if total_cost > balance:
            return "Insufficient balance to buy!"
        balance = balance - total_cost
        if symbol in portfolio:
            portfolio[symbol] += shares
        else:
            portfolio[symbol] = shares
        print("[BUY] After:", portfolio)
        return f"Bought {shares} shares of {symbol} at ₹{price:.2f} each.\nRemaining balance: ₹{balance:.2f}"
    except Exception as e:
        print("[BUY] Error:", e)
        return "Error: " + str(e)

def sell_stock(portfolio, symbol, shares):
    global balance
    symbol = symbol.upper()  # Always use uppercase, no .NS in portfolio
    print("[SELL] Before:", portfolio)
    if symbol not in portfolio:
        return "You don't own any shares of " + symbol
    if portfolio[symbol] < shares:
        return "You don't have enough shares to sell!"
    stock = yf.Ticker(symbol + ".NS")
    price = stock.history(period="1d")["Close"].iloc[-1]
    balance = balance + (price * shares)
    portfolio[symbol] -= shares
    print("[SELL] After:", portfolio)
    return f"Sold {shares} shares of {symbol} at ₹{price:.3f} each.\nRemaining balance: ₹{balance:.3f}"

def get_stock_prices(tickers):
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            if not hist.empty and 'Close' in hist.columns:
                v = hist['Close'].iloc[-1]
                data[ticker] = f"{v:.2f}"
            else:
                data[ticker] = "N/A"
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            data[ticker] = "N/A"
    return data

def google_search(query):
    try:
        url = f"https://www.google.com/search?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(url, headers=headers) #Sends the request to get the information from the google
        response.raise_for_status()  
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd') #We got the class code by inspecting the google search page
        return results[0].get_text() if results else "No results found"
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"

# Authentication Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose a different username.', 'error')
            return render_template('register.html')
        
        # Create new user
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('landing'))

# Routes for different pages: index, portfolio, buy, sell, latest_news, analyze
# The index route displays the homepage with scrolling live stock prices
# The portfolio route shows the user's portfolio and its performance
# The buy route allows users to buy stocks
# The sell route allows users to sell stocks
# The latest_news route displays the latest business news
# The analyze route analyzes a stock's performance and displays relevant charts

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/')
def index():
    if current_user.is_authenticated:
        # Get prices for some popular Indian stocks
        tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
        stock_prices = get_stock_prices(tickers)
        
        # Check if user is authenticated to show user info
        user = current_user if current_user.is_authenticated else None
        return render_template('index.html', stock_prices=stock_prices, user=user)
    else:
        return redirect(url_for('landing'))

@app.route('/portfolio')
@login_required
def view_portfolio():
    # Get user's portfolio from database
    user_portfolio = Portfolio.query.filter_by(user_id=current_user.id).all()
    
    # Convert to dictionary format for existing display function
    portfolio_dict = {}
    for item in user_portfolio:
        if item.symbol in portfolio_dict:
            portfolio_dict[item.symbol] += item.shares
        else:
            portfolio_dict[item.symbol] = item.shares
    
    portfolio_info = display_portfolio(portfolio_dict, current_user.balance)
    return render_template('portfolio.html', portfolio_info=portfolio_info, user=current_user)

@app.route('/buy', methods=['GET', 'POST'])
@login_required
def buy():
    if request.method == 'POST':
        symbol = request.form['symbol'].upper()
        shares = int(request.form['shares'])
        print(f"[BUY] Attempting to buy {shares} shares of {symbol}")
        
        # Get current stock price
        try:
            stock = yf.Ticker(symbol + ".NS")
            price = stock.history(period="1d")["Close"].iloc[-1]
            total_cost = price * shares
            
            if total_cost > current_user.balance:
                flash("Insufficient balance to buy!", 'error')
                return render_template('buy.html', message="Insufficient balance", symbol=symbol, shares=shares)
            
            # Update user balance
            current_user.balance -= total_cost
            
            # Add to portfolio
            new_portfolio_item = Portfolio(
                user_id=current_user.id,
                symbol=symbol,
                shares=shares,
                avg_price=price
            )
            db.session.add(new_portfolio_item)
            db.session.commit()
            
            message = f"Bought {shares} shares of {symbol} at ₹{price:.2f} each.\nRemaining balance: ₹{current_user.balance:.2f}"
            flash(message, 'success')
            return render_template('buy.html', message=message, symbol=symbol, shares=shares)
            
        except Exception as e:
            flash(f"Error: {str(e)}", 'error')
            return render_template('buy.html')
    
    return render_template('buy.html')

@app.route('/confirm_buy', methods=['POST'])
@login_required
def confirm_buy():
    symbol = request.form['symbol'].upper()
    shares = int(request.form['shares'])
    print(f"[CONFIRM BUY] Attempting to buy {shares} shares of {symbol}")
    
    try:
        stock = yf.Ticker(symbol + ".NS")
        price = stock.history(period="1d")["Close"].iloc[-1]
        total_cost = price * shares
        
        if total_cost > current_user.balance:
            flash("Insufficient balance to buy!", 'error')
            return render_template('message.html', message="Insufficient balance")
        
        # Update user balance
        current_user.balance -= total_cost
        
        # Add to portfolio
        new_portfolio_item = Portfolio(
            user_id=current_user.id,
            symbol=symbol,
            shares=shares,
            avg_price=price
        )
        db.session.add(new_portfolio_item)
        db.session.commit()
        
        message = f"Bought {shares} shares of {symbol} at ₹{price:.2f} each.\nRemaining balance: ₹{current_user.balance:.2f}"
        flash(message, 'success')
        return render_template('message.html', message=message)
        
    except Exception as e:
        flash(f"Error: {str(e)}", 'error')
        return render_template('message.html', message=f"Error: {str(e)}")

@app.route('/sell', methods=['GET', 'POST'])
@login_required
def sell():
    if request.method == 'POST':
        symbol = request.form['symbol'].upper()
        shares = int(request.form['shares'])
        print(f"[SELL] Attempting to sell {shares} shares of {symbol}")
        
        # Check if user has enough shares
        user_shares = Portfolio.query.filter_by(user_id=current_user.id, symbol=symbol).all()
        total_shares = sum(item.shares for item in user_shares)
        
        if total_shares < shares:
            flash("You don't have enough shares to sell!", 'error')
            return render_template('sell.html', message="Insufficient shares", symbol=symbol, shares=shares)
        
        try:
            stock = yf.Ticker(symbol + ".NS")
            price = stock.history(period="1d")["Close"].iloc[-1]
            total_value = price * shares
            
            # Update user balance
            current_user.balance += total_value
            
            # Remove shares from portfolio (FIFO method)
            remaining_shares = shares
            for item in user_shares:
                if remaining_shares <= 0:
                    break
                if item.shares <= remaining_shares:
                    remaining_shares -= item.shares
                    db.session.delete(item)
                else:
                    item.shares -= remaining_shares
                    remaining_shares = 0
            
            db.session.commit()
            
            message = f"Sold {shares} shares of {symbol} at ₹{price:.2f} each.\nNew balance: ₹{current_user.balance:.2f}"
            flash(message, 'success')
            return render_template('sell.html', message=message, symbol=symbol, shares=shares)
            
        except Exception as e:
            flash(f"Error: {str(e)}", 'error')
            return render_template('sell.html')
    
    return render_template('sell.html')

@app.route('/confirm_sell', methods=['POST'])
@login_required
def confirm_sell():
    symbol = request.form['symbol'].upper()
    shares = int(request.form['shares'])
    print(f"[CONFIRM SELL] Attempting to sell {shares} shares of {symbol}")
    
    # Check if user has enough shares
    user_shares = Portfolio.query.filter_by(user_id=current_user.id, symbol=symbol).all()
    total_shares = sum(item.shares for item in user_shares)
    
    if total_shares < shares:
        flash("You don't have enough shares to sell!", 'error')
        return render_template('message.html', message="Insufficient shares")
    
    try:
        stock = yf.Ticker(symbol + ".NS")
        price = stock.history(period="1d")["Close"].iloc[-1]
        total_value = price * shares
        
        # Update user balance
        current_user.balance += total_value
        
        # Remove shares from portfolio (FIFO method)
        remaining_shares = shares
        for item in user_shares:
            if remaining_shares <= 0:
                break
            if item.shares <= remaining_shares:
                remaining_shares -= item.shares
                db.session.delete(item)
            else:
                item.shares -= remaining_shares
                remaining_shares = 0
        
        db.session.commit()
        
        message = f"Sold {shares} shares of {symbol} at ₹{price:.2f} each.\nNew balance: ₹{current_user.balance:.2f}"
        flash(message, 'success')
        return render_template('message.html', message=message)
        
    except Exception as e:
        flash(f"Error: {str(e)}", 'error')
        return render_template('message.html', message=f"Error: {str(e)}")

@app.route('/update_prices')
def update_prices():
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS', 'TATAMOTORS.NS', 'MRF.NS', 'TCS.NS', "HSCL.NS"]
    stock_prices = get_stock_prices(tickers)
    return jsonify(stock_prices)

@app.route('/stream')
def stream():
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS', 'TATAMOTORS.NS', 'MRF.NS', 'TCS.NS', "HSCL.NS"]
    def event_stream():
        while True:
            yield 'data: {}\n\n'.format(jsonify(get_stock_prices(tickers)))
            time.sleep(10)  # Update after every 10 seconds 
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/latest_news')
def latest_news():
    news = get_news()
    print("NEWS FETCHED:", news)  # Debug print
    user = current_user if current_user.is_authenticated else None
    return render_template('stock_news.html', news=news, user=user)

@app.route('/ai-predict')
def ai_predict():
    user = current_user if current_user.is_authenticated else None
    return render_template('ai_predict.html', user=user)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        print("=== AI ANALYSIS STARTED ===")
        symbol = request.form['symbol'].upper()
        print(f"Symbol received: {symbol}")
        
        # Initialize the AI predictor
        ai_predictor = StockAI()
        
        # Get prediction using the AI predictor
        prediction_result = ai_predictor.predict_stock(symbol)
        
        print("Analysis completed successfully!")
        print(f"Prediction: {prediction_result}")
        
        return render_template('ai_predict.html', prediction=prediction_result)
        
    except Exception as e:
        print(f"ERROR in analyze: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f'Error analyzing stock: {str(e)}', 'error')
        return render_template('ai_predict.html', prediction=None)

@app.route('/get_stock_price/<symbol>')
def get_stock_price(symbol):
    """Get current stock price for a given symbol"""
    try:
        # Add .NS suffix for Indian stocks if not present
        if not symbol.endswith('.NS'):
            symbol_with_suffix = f"{symbol}.NS"
        else:
            symbol_with_suffix = symbol
        
        stock = yf.Ticker(symbol_with_suffix)
        hist = stock.history(period="1d")
        
        if not hist.empty and 'Close' in hist.columns:
            price = hist['Close'].iloc[-1]
            return jsonify({
                'success': True,
                'price': float(price),
                'symbol': symbol
            })
        else:
            return jsonify({
                'success': False,
                'error': f'No price data available for {symbol}'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error fetching price for {symbol}: {str(e)}'
        })

@app.route('/check_user_shares/<symbol>')
@login_required
def check_user_shares(symbol):
    """Check how many shares a user has for a given symbol"""
    try:
        user_shares = Portfolio.query.filter_by(user_id=current_user.id, symbol=symbol.upper()).all()
        total_shares = sum(item.shares for item in user_shares)
        
        return jsonify({
            'success': True,
            'shares': total_shares,
            'symbol': symbol.upper()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error checking shares for {symbol}: {str(e)}'
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5003))
    app.run(host='0.0.0.0', port=port, debug=False)
















