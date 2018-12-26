library(tidyverse)
library(ggfortify)
library(lubridate)
library(scales)
library(geosphere)
library(glmnet)

#### Import and clean ####

# Set a maximum import size

import_size <- 10000000L

# Import and clean the data

datafile <- 'data/train.csv'
taxi_data <- read_csv(datafile, n_max = import_size) %>%
  select(-pickup_datetime) %>%
  rename(pickup_datetime = key) %>%
  filter(fare_amount>0 &
          pickup_latitude>40.5 & pickup_latitude<41 &
          pickup_longitude> -74.5 & pickup_longitude< -73.5 &
          dropoff_latitude>40.5 & dropoff_latitude<41 &
          dropoff_longitude> -74.5 & dropoff_longitude< -73.5)

#### Exploratory visualisations ####

# Set theme details

blog_colour <- "#fdfdfd"

theme_set(theme_classic() + 
            theme(plot.background = element_rect(fill = blog_colour),
                  panel.background = element_rect(fill = blog_colour),
                  legend.background = element_rect(fill = blog_colour)))

# Histogram of fares

ggplot(taxi_data, aes(fare_amount)) +
  geom_histogram(colour = 'black', fill = 'white', breaks = seq(0,100,2), aes(y = ..density..)) +
  scale_x_continuous(labels = dollar, breaks = seq(0,100,10), limits = c(0,100)) +
  ggtitle('Distribution of fares') +
  xlab('Fare amount') +
  ylab('Density')

ggsave('fares_distribution.png', path = 'output')

# Monthly mean fares

taxi_data %>%
  mutate(pickup_month = floor_date(pickup_datetime, 'month')) %>%
  group_by(pickup_month) %>%
  summarise(mean_fare = mean(fare_amount)) %>%
  ggplot(aes(pickup_month, mean_fare)) +
  geom_line() +
  scale_y_continuous(labels = dollar) +
  ggtitle('Mean fares by month') +
  xlab('Pickup month') +
  ylab('Mean fare')

ggsave('monthly_mean_fares.png', path='output/')

# Plot of pickup locations

ggplot(sample_frac(taxi_data, 0.01), aes(x = pickup_longitude, y = pickup_latitude)) +
  geom_point(shape = '.', alpha = 0.5) +
  xlim(-74.1, -73.75) +
  ylim(40.6, 40.9) +
  coord_equal() +
  ggtitle('Pickup locations') +
  xlab('Pickup longitude') +
  ylab('Pickup latitude')

ggsave('pickup_coordinates.png', path = 'output/')

#### Linear model ####

# Perform some further initial processing

processed_taxi_data <- taxi_data %>%
  mutate(pickup_date = as.integer(date(pickup_datetime)),
         pickup_month = month(pickup_datetime, label=TRUE),
         pickup_weekday = wday(pickup_datetime, label=TRUE),
         pickup_hour = hour(pickup_datetime)) %>%
  filter(date(pickup_datetime) > as_date('2012-09-01')) %>%
  mutate(haversine = distHaversine(cbind(pickup_longitude, pickup_latitude),
                               cbind(dropoff_longitude, dropoff_latitude)),
       bearing = bearing(cbind(pickup_longitude, pickup_latitude),
                       cbind(dropoff_longitude, dropoff_latitude)),
       bearing_adj = bearing-29,
       bearing_rad = abs(bearing_adj * pi / 180),
       manhattan = haversine*abs(sin(bearing_rad))+haversine*abs(cos(bearing_rad))
)

# Histogram of calculated distances

processed_taxi_data %>%
  select(haversine, manhattan) %>%
  gather(value = "distance", key = "metric") %>%
  sample_frac(size = 0.2) %>%
  ggplot(aes(x = distance)) +
  geom_histogram(colour = 'black', fill = 'white', aes(y = ..density..),
                 breaks = seq(0,8000,100)) +
  facet_grid(metric ~ .) +
  labs(title = 'Comparison of distance metrics', x = 'Distance (m)', y = 'Density')

ggsave('metric_comparisons.png', path = 'output/')

# Now we can build and plot our initial model

lm_manhattan <- lm(fare_amount ~ manhattan, data = processed_taxi_data)

summary(lm_manhattan)

ggplot(sample_frac(processed_taxi_data, size = 0.001), 
       aes(x = manhattan, y = fare_amount)) +
  geom_point(alpha = 0.3, size = 1) +
  ggtitle('Fare amount on distance') +
  xlab('Distance (m)') +
  ylab('Fare amount') +
  scale_y_continuous(labels = dollar, breaks = seq(0,120,20), limits = c(0,120)) +
  geom_abline(intercept = lm_manhattan$coefficients[[1]], 
              slope = lm_manhattan$coefficients[[2]],
              colour = 'red', linetype = 'longdash')

ggsave('manhattan_vs_fare.png', path='output/')

# Add in some superfluous variables

lm_generalised <- lm(fare_amount ~ manhattan + passenger_count + bearing,
                     data = processed_taxi_data)

summary(lm_generalised)

# And try again with even more superfluous methods

lm_even_generaliseder <- lm(fare_amount ~ manhattan + passenger_count + 
                              bearing + pickup_latitude + pickup_longitude +
                              dropoff_latitude + dropoff_longitude,
                     data = processed_taxi_data)

summary(lm_even_generaliseder)

# Clear memory

rm(lm_manhattan, lm_generalised, lm_even_generaliseder)

#### Shrinkage methods ####

design_matrix <- processed_taxi_data %>%
  select(fare_amount,
         manhattan, passenger_count, bearing,
         pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude) %>%
  sample_frac(size = 0.1) %>%
  as.matrix()

response <- design_matrix[,1]
response_scaled <- scale(response)
design_matrix <- design_matrix[,2:8]
design_matrix_scaled <- scale(design_matrix)

# LASSO

model <- cv.glmnet(x = design_matrix_scaled, y = response_scaled, alpha = 1)

coefs <- coef(model, s = "lambda.1se")
print(coefs)

unscaled_coefs <- scaleBack.lm(design_matrix, response, 
                               coefs %>% as.vector() %>% as.matrix())
print(unscaled_coefs[1:2])

autoplot(model) +
  labs(title = 'Cross-validation of LASSO fit') +
  scale_y_continuous(labels = percent)

ggsave('cross-validation_LASSO.png', path = 'output/')

autoplot(model$glmnet.fit) +
  labs(title = 'Profile of LASSO coefficients')

ggsave('coefficient_profiles_LASSO.png', path = 'output/')

rm(model, coefs, unscaled_coefs)

# Ridge regression

model <- cv.glmnet(x = design_matrix_scaled, y = response_scaled, alpha = 0)

coefs <- coef(model, s = "lambda.1se")
print(coefs)

unscaled_coefs <- scaleBack.lm(design_matrix, response, 
                               coefs %>% as.vector() %>% as.matrix())
print(unscaled_coefs[1:2])

autoplot(model) +
  labs(title = 'Cross-validation of ridge regression fit') +
  scale_y_continuous(labels = percent)

ggsave('cross-validation_ridge.png', path = 'output/')

autoplot(model$glmnet.fit) +
  labs(title = 'Profile of ridge regression coefficients')

ggsave('coefficient_profiles_ridge.png', path = 'output/')

rm(model, coefs, unscaled_coefs)

# Elastic net

model <- cv.glmnet(x = design_matrix_scaled, y = response_scaled, alpha = 0.5)

coefs <- coef(model, s = "lambda.1se")
print(coefs)

unscaled_coefs <- scaleBack.lm(design_matrix, response, 
                               coefs %>% as.vector() %>% as.matrix())
print(unscaled_coefs[1:2])

autoplot(model) +
  labs(title = 'Cross-validation of elastic-net fit') +
  scale_y_continuous(labels = percent)

ggsave('cross-validation_elastic-net.png', path='output/')

autoplot(model$glmnet.fit) +
  labs(title = 'Profile of elastic-net coefficients')

ggsave('coefficient_profiles_elastic-net.png', path = 'output/')

rm(model, coefs, unscaled_coefs)

