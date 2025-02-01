To build a demand forecasting model based on master data and sales data, taking into account various features, you can follow these steps:

### 1. **Data Collection and Preprocessing**:
   - **Master Data**: Collect relevant master data, such as:
     - Product details (SKU, category, brand, price, etc.)
     - Time features (seasonality, promotional periods, holidays, etc.)
     - Inventory data (stock levels, replenishment schedules, etc.)
   - **Sales Data**: Collect historical sales data, including:
     - Sales volume (daily, weekly, or monthly)
     - Transaction details (quantity sold, date of sale, etc.)
     - External factors (weather, economic conditions, etc.)

### 2. **Feature Engineering**:
   - **Time-based Features**: Extract features like:
     - Day of the week, month, year
     - Holidays, weekends, seasonality
   - **Product-based Features**: 
     - Price elasticity (how sales change with price)
     - Product lifecycle (e.g., new, mature, discontinued)
     - Promotional campaigns
   - **External Features**: 
     - Weather conditions (if relevant for your product type)
     - Economic indicators (inflation, market trends)
     - Competitor actions or market events

### 3. **Model Selection**:
   - **Statistical Models**: Start with traditional methods like ARIMA, SARIMA for time series forecasting.
   - **Machine Learning Models**: Consider using models like:
     - Random Forests
     - Gradient Boosting Machines (e.g., XGBoost, LightGBM)
     - Neural Networks (e.g., LSTM for sequential data)
   - **Deep Learning Models**: Use advanced architectures like:
     - Recurrent Neural Networks (RNN)
     - Long Short-Term Memory Networks (LSTM)
     - Prophet (for capturing seasonality and trend)

### 4. **Model Training and Validation**:
   - Split the data into training and test sets (typically use cross-validation).
   - Train the model on historical sales data with the engineered features.
   - Tune hyperparameters to improve model performance (e.g., grid search, random search).
   - Validate the model using accuracy metrics like RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), or MAPE (Mean Absolute Percentage Error).

### 5. **Demand Forecasting**:
   - Use the trained model to predict demand for the next 14 days (daily forecast for each product, SKU).
   - Incorporate seasonality, promotions, and other relevant features to improve accuracy.

### 6. **Model Deployment and Monitoring**:
   - Deploy the model into a production environment where it can forecast future demand.
   - Monitor the model’s performance over time and retrain it periodically with updated sales data to ensure its accuracy.

By using master data and sales data along with various relevant features, this approach will help generate accurate demand forecasts for your products over a 14-day horizon.