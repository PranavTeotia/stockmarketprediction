<?php
session_start();
include("connect.php");

?>
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TradeX</title>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <link rel="stylesheet" href="style.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="styles.css?v=<?php echo time(); ?>">

</head>

<body>
    <!-- Navbar -->
<style>
    body{
        overflow-x:hidden;

    }
    *{
        text-decoration: none;
    }
    .navbar{
        /* background-color: rgb(19, 196, 196);
        */
        background-color:transparent;
        font-family: sans-serif;
        padding-right: 15px;
        padding-left: 15px;
        
    }
    .navdiv{
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .logo a{
        font-size: 35px;
        font-weight: 600;
        color: black;
    }
    li{
        list-style: none;
        display: inline-block;
    }
    li a{
        color: black;
        font-size: 18px;
        font-weight: bold;
        margin-right: 25px;
    }

    /* Add padding to body to prevent content overlap */
    .content {
        padding-top: 60px; /* Match the height of the navbar */
    }
    /* Dropdown button */
    .dropdown {
        position: relative;
        display: inline-block;
    }

    .dropbtn {
        /* background-color: #4CAF50;  */
        color: black;    
        border: none;
        font-size: 18px;
        font-weight: bold;
        margin-right: 25px;
        cursor: pointer;
    }

    /* Dropdown Content (hidden by default) */
    .dropdown-content {
        display: none;
        position: absolute;
        background-color: #f9f9f9;
        min-width: 160px;
        box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.2);
        z-index: 1;
    }

    .dropdown-content a {
        color: black;
        padding: 8px 10px;
        text-decoration: none;
        display: block;
    }

    /* Show the dropdown when hovered */
    .dropdown:hover .dropdown-content {
        display: block;
    }

    .dropdown-content a:hover {
        background-color: #ddd; /* Change background on hover */
    }
</style>

<nav class="navbar">
        <div class="navdiv">
            <div class="logo"><a>TradeX</a></div>
            <ul>
            <li><a href="#">Home</a></li>
            <li class="dropdown">
            <a href="javascript:void(0);" class="dropbtn">Service</a>
            <div class="dropdown-content">
                <a href="#lstm">Price Prediction</a>
                <a href="#snta">Trend Prediction </a>
                <a href="#News1">Sentiment Analysis</a>
            </div>
            </li>
            <li><a href="#News">News</a></li>
            <li><a href="#contact">Contact</a></li>
            <li><a href="logout.php">Logout</a></li>        
            </ul>
        </div>
        
    </nav>
    
    
<!-- 1st page -->
 <style>
.slider {

    justify-content: center;
    align-items: center;
    height: 730px;
    background-repeat: no-repeat;
    background-size: cover;
    background-position: center;
    
    animation: slider 7s infinite linear;
    background-image: url('austin-distel-nGc5RT2HmF0-unsplash.jpg');

}

@keyframes slider {
    0% {
        background-image: url('anne-nygard-tcJ6sJTtTWI-unsplash.jpg');
    }

    20% {
        background-image: url('clay-banks-3IFYE6UHFBo-unsplash.jpg');
    }

    35% {
        background-image: url('austin-distel-nGc5RT2HmF0-unsplash.jpg');
    }

    40% {
        background-image: url('markus-spiske-XrIfY_4cK1w-unsplash.jpg');
    }

    50% {
        background-image: url('jason-briscoe-Gw_sFen8VhU-unsplash.jpg');
    }

    70% {
        background-image: url('markus-spiske-jgOkEjVw-KM-unsplash.jpg');
    }

    100% {
        background-image: url('dylan-calluy-j9q18vvHitg-unsplash.jpg');
    }
}
 </style>
<div class="slider" >
</div>
<style>
/* Style for the chart container */

    .chart-container {
        display: flex;
        justify-content:center; 
        gap: 20px; /* Add space between charts */
        margin: 20px; /* Add space around the charts */
        flex-wrap: wrap;/* Allow wrapping on smaller screens */
    }

    /* Individual chart styling */
    .chart {
        width: 47%; /* Adjust width to fit side by side */
        height: 700px;
        border: 1px solid #ddd; /* Optional: Add a border for better visualization */
        border-radius: 8px; /* Optional: Add rounded corners */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Optional: Add shadow */
        background-color: #fff; /* Ensure a consistent background */
    }

    /* Responsive design for smaller screens */
    @media (max-width: 768px) {
        .chart {
            width: 90%; /* Full width for small screens */
            margin-bottom: 20px; /* Add space between rows */
        }

    }
    h4{
        color: black;
        text-align: center;
        font-size:30px;
        margin-bottom:0px;
        margin-top:0px;
        
    }
</style>
<div class="content" id="btns">
    <h4>Stock Charts</h4>   

    <div class="chart-container">

        <!-- First Chart -->
        <div id="chart1" class="chart"></div>

        <!-- Second Chart -->
        <div id="chart2" class="chart"></div>
        <!-- <div id="chart3" class= "chart"></div> -->
    </div>
</div>
    <script type="text/javascript">
        // First Stock Chart
        new TradingView.widget({
            "container_id": "chart1",
            "symbol": "NASDAQ:AAPL", // Replace with your first stock symbol
            "interval": "D",
            "theme": "light",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#f1f3f6",
            "allow_symbol_change": true, // Enables symbol search functionality

            "enable_publishing": false,
            "hide_side_toolbar": false,
            "details": true,
            "withdateranges": true, // Allows date range selection
            "show_popup_button": true,
            "studies": [],
            "autosize": true
        });

        // Second Stock Chart
        new TradingView.widget({
            "container_id": "chart2",
            "symbol": "NASDAQ:GOOGL", // Replace with your second stock symbol
            "interval": "D",
            "theme": "light",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": false,
            "hide_side_toolbar": false,
            "allow_symbol_change": true, // Enables symbol search functionality
            "details": true,
            "withdateranges": true, // Allows date range selection
            "show_popup_button": true,
            "studies": [],
            "autosize": true
        });
        
    </script>
    <div class="full-page-container" id="lstm">
        <iframe src="http://localhost:5000" width="100%" height="800px" style="border:none;"></iframe>
    </div>
    
    <div class="full-page-container" id="snta">
        <iframe src="http://localhost:5001" width="100%" height="800px" style="border:none;"></iframe>
    </div>
    <div class="full-page-container" id="News1">
        <iframe src="http://localhost:5002" width="100%" height="800px" style="border:none;"></iframe>
    </div>
    
    <div class="tradingview-widget-container" id="News">
    <div class="tradingview-widget-container__widget"></div>
    <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-timeline.js">
    {
        "feedMode": "market",
        "colorTheme": "light",
        "isTransparent": false,
        "displayMode": "compact",
        "width": "100%",
        "height": "800"
    }
    </script>

    </div>

<style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f9f9f9;
            color: #333;
        }

        .section {
            padding: 50px;
            background-color: #fff;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            /* margin: 20px; */
            border-radius: 8px;
        }

        .section h2 {
            font-size: 2rem;
            margin-bottom: 20px;
            color: #0078d7;
        }

        .contact-form, .social-map-container {
            margin: 20px 0;
        }

        .social-map-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }

        .social-media, .map, .about-section {
            flex: 1;
            min-width: 300px;
            

        }

        .social-media {
            margin-top: 10px;

        }

        .social-media a {
            display:block;
            text-decoration: none;
            color: #fff;
            margin-bottom:10px;
            padding: 10px 15px;
            border-radius: 5px;
            width: 150px;
        }

        .social-media a.facebook { background-color: #1da1f2;}
        .social-media a.twitter { background-color:  #1da1f2;}
        .social-media a.linkedin { background-color: #0077b5; }

        .map iframe {
            width: 100%;
            height: 300px;
            border: none;
            border-radius: 8px;
        }

        .contact-form form {
            display: flex;
            flex-direction: column;
        }

        .contact-form label {
            margin-bottom: 5px;
            font-weight: bold;
        }

        .contact-form input, .contact-form textarea, .contact-form button {
            margin-bottom: 15px;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 1rem;
        }

        .contact-form button {
            background-color: #0078d7;
            color: white;
            border: none;
            cursor: pointer;
        }

        .contact-form button:hover {
            background-color: #005bb5;
        }

        .about-section {
            padding: 20px;
            /* background-color: #f1f1f1; */
            border-radius: 8px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }
        

        @media (max-width: 768px) {
            .section {
                padding: 20px;
            }

            .social-map-container {
                flex-direction: column;
            }
        }
    </style>


<div class="section">
    <!-- <h2>Contact Us</h2> -->
    <div class="contact-form">
        <h3>Get in Touch</h3>
        <form action="https://formsubmit.co/prakharteotia77@gmail.com" method="POST">
            <label for="name">Name:</label>
            <input type="text" id="name" name="name" placeholder="Your name" required>

            <label for="email">Email:</label>
            <input type="email" id="email" name="email" placeholder="Your email" required>

            <label for="message">Message:</label>
            <textarea id="message" name="message" rows="5" placeholder="Your message" required></textarea>

            <!-- Anti-spam field -->
            <input type="hidden" name="_captcha" value="false">

            <button type="submit">Submit</button>
        </form>
    </div>
    <div class="social-map-container">
    <div class="about-section">
            <h3>About Us</h3>
            <p>
                Welcome to our Stock Market Prediction platform! Our mission is to leverage cutting-edge technologies like LSTM models and news sentiment analysis to provide accurate stock market predictions. We aim to empower investors with data-driven insights for better decision-making.
            </p>
            <p>
                Explore our website to view predicted stock prices, trends, and sentiment scores, all powered by reliable data from yfinance and our advanced models.
            </p>
        </div>
        <div class="social-media">
            <h3>Connect with Us</h3>
            <a href="#" class="facebook">Facebook</a>
            <a href="#" class="twitter">Twitter</a>
            <a href="#" class="linkedin">LinkedIn</a>
        </div>
        <div class="map">
            <h3>Our Location</h3>
            <iframe 
            src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3484.725715176126!2d77.63842831506404!3d28.973047079394026!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x390c668fdea4d87f%3A0x8795def814a486e7!2sMeerut%20Institute%20of%20Engineering%20and%20Technology!5e0!3m2!1sen!2sin!4v1703241234567!5m2!1sen!2sin" 
            allowfullscreen="" 
                loading="lazy">
            </iframe>
        </div>
        
    </div>
</div>
 
<script src="script.js"></script>
</body>
</html>
