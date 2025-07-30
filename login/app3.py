from flask import Flask, request, render_template, Response
import yfinance as yf
import pandas as pd
from yahoo_fin import news
from transformers import pipeline
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Set Matplotlib to non-interactive backend
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)
# Initialize the FinBERT sentiment analysis pipeline
_sentiment_analysis = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Define the home route
@app.route('/')
def index():
    # return 'hi'
    return render_template('index3.html')  # Simple HTML form to accept ticker input

# Define the route for sentiment analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    print("Analyze route hit!")  # Debug print
    # return render_template('results.html')

    # Get the ticker symbol from the form
    ticker = request.form.get('ticker', 'TSLA')  # Default to TSLA if not provided

    # Fetch news data for the ticker
    news_data = news.get_yf_rss(ticker)
    news_df = pd.DataFrame(news_data)

    # Check if data is available
    if news_df.empty:
        return "No news data found for the provided ticker."

    # Process the news DataFrame
    df = news_df[['title', 'link', 'published']].copy()
    df.rename(columns={'title': 'Headline', 'link': 'Link', 'published': 'DateTime'}, inplace=True)
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    # Perform sentiment analysis
    df['finbert_sentiment'] = df['Headline'].apply(_sentiment_analysis)
    df['finbert_sentiment_score'] = df['finbert_sentiment'].apply(
        lambda x: {'negative': -1, 'positive': 1}.get(x[0]['label'].lower(), 0) * x[0]['score']
    )
    df['finbert_sentiment_score'] = df['finbert_sentiment_score'].round(2)

    # Sort data by DateTime
    df.sort_values('DateTime', inplace=True)

    plt.figure(figsize=(7, 4))
    plt.plot(df['DateTime'], df['finbert_sentiment_score'], marker='o', linestyle='-', color='b')
    plt.title('FinBERT Sentiment Score Over Time')
    plt.xlabel('DateTime')
    plt.ylabel('Sentiment Score')
    plt.tight_layout()
    plot_path = 'login/static/sentiment_plot.png'
    plt.savefig(plot_path)  # Save the plot
    plt.close()  # Close the figure to avoid resource warnings


    # Render the results template
    return render_template('result3.html', plot_path=plot_path,news=df[['Headline', 'Link','finbert_sentiment_score']].head(10).to_dict(orient='records'))
    
    
    

# Run the app
if __name__ == '__main__':
    app.run(port=5002, debug=True)
