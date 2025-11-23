##* The Bayesian Vector Autoregression model is an expansion of the 
##* normal VAR model that treats the estimated parameters as random variables 
##* instead of fixed values. This generally shrinks parameter estimates, 
##* and is helpful in giving more stable estimates in smaller datasets. 
##* 
##* Since our data is not very high frequency, and doesn't go back very far 
##* (quarterly going back to 2011), this modification to the VAR model 
##* makes sense for our purposes (and is in line with other popular GDP forecasting models). 

# =========================== PACKAGES =========================================
library(dplyr) # for data manipulation
library(BVAR) # for modelling BVAR
library(xts) # for time series object
library(ggplot2) # for plotting



# ============================ DATA ============================================
data_raw <- read.csv("../data/data_wpk.csv")

# drop nominal GDP (we'll forecast growth)
data <- data_raw %>% select(-gdp) 
data$date <- as.Date(data$date)

# filter data by dat
data_2011 <- data %>% filter(date >= "2011-01-01", date <= "2025-06-30")

# convert to time series object
data_xts <- xts(data_2011[,-1], order.by = data_2011$date)

# make sure data is complete
sum(is.na(data_xts))

# too many variables, dropping orders (since it's range is tiny)
data_1978 <- data %>% select(-orders, -construction, -itrade, -wtrade) %>% 
  filter(date >= "1978-01-01", date <= "2025-06-30")

data_xts <- xts(data_1978[,-1], order.by = data_1978$date)
sum(is.na(data_xts))

# really limiting our predictors:
data_less_vars <- data_raw %>% select(date, gdp_yoy, employment, unemploy_claims, bus_outlook) %>% 
  mutate(date = as.Date(date)) %>% filter(date >= "1968-06-30", date <= "2025-06-30")
sum(is.na(data_less_vars))

data_xts_less_vars <- xts(data_less_vars[,-1], order.by = data_less_vars$date)

data_xts_logged <- data_xts_less_vars
for (col in c("employment", "unemploy_claims")) {
  data_xts_logged[, col] <- log(data_xts_logged[, col])
}

# =========================== MODELING =========================================
# select lags
lags <- 4  # start with 4, typical for quarterly data

mn <- bv_minnesota(lambda = bv_lambda(mode = 0.2, sd = 0.4, min = 0.0001, max = 5),
                   alpha = bv_alpha(mode = 2), var = 1e07)

soc <- bv_soc(mode = 1, sd = 1, min = 1e-04, max = 50)
sur <- bv_sur(mode = 1, sd = 1, min = 1e-04, max = 50)

priors <- bv_priors(hyper = "auto", mn = mn, soc = soc, sur = sur)

mh <- bv_metropolis(scale_hess = c(0.05, 0.0001, 0.0001),adjust_acc = TRUE, acc_lower = 0.25, acc_upper = 0.45)

run <- bvar(data_xts_logged, lags = 1, n_draw = 50000, n_burn = 25000, n_thin = 1,
            priors = priors, mh = mh, verbose = TRUE)

x <- fred_qd[1:243, c("GDPC1", "PCECC96", "GPDIC1",
                      "HOANBS", "GDPCTPI", "FEDFUNDS")]

# estimate BVAR model
model <- bvar(
  data_xts_less_vars,
  lags = 2,
  n_draw = 5000,
  n_burn = 1000,
  priors = priors
)

plot(model)
plot(residuals(model, type = "mean"))

# get nowcast
forecast <- predict(model, horizon = 16, conf_bands = c(0.05, 0.16))

# Extract GDP nowcast
nowcast_gdp <- forecast$mean[,"gdp_yoy"]  # adjust name as needed
print(nowcast_gdp)

# intervals
forecast$interval[,"gdp_yoy",]

# ========================== VISUALIZATION =====================================

plot(model)
plot(residuals(model, type = "mean"))
plot(irf(model))

plot(forecast)
plot(forecast, area = TRUE, vars = "gdp_yoy")


# Last historical GDP point
last_date <- index(data_xts)[NROW(data_xts)]
next_date <- last_date + 90   # rough for quarterly; adjust for freq

# Build a small forecast ts object
fc_df <- data.frame(
  date = c(last_date, next_date),
  GDP = c(last(data_xts$GDP), nowcast_gdp)
)

ggplot() +
  geom_line(data = fortify.zoo(data_xts)[,c("Index","GDP")],
            aes(Index, GDP), size=1) +
  geom_point(data = fc_df, aes(date, GDP), color="red", size=3) +
  geom_line(data = fc_df, aes(date, GDP), color="red", linetype="dashed") +
  labs(title="GDP Nowcast (BVAR)',
       x="Date", y="GDP") +
  theme_minimal()
