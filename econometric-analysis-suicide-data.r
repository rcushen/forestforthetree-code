library(tidyverse)
library(ggfortify)

blog_colour <- "#fdfdfd"
theme_set(theme_classic() + 
            theme(plot.background = element_rect(fill = blog_colour),
                  panel.background = element_rect(fill = blog_colour),
                  legend.background = element_rect(fill = blog_colour)))

options(warn=-1)

suicide_data <- read_csv('master.csv') %>%
    mutate(age=factor(age, levels=c('5-14 years', '15-24 years', '25-34 years', 
                                    '35-54 years', '55-74 years', '75+ years')),
           generation=factor(generation, levels=c('G.I. Generation', 'Silent', 'Boomers',
                                                  'Generation X', 'Millenials')))

#### The Dataset ####

glimpse(suicide_data)

suicide_data %>%
    group_by(year) %>%
    summarise(n_countries=n_distinct(country)) %>%
    ggplot(aes(x=year, y=n_countries)) +
        geom_col(color='black', fill='white') +
        xlab('Value') + ylab('Number of countries present') +
        ggtitle('Available countries per year')

ggsave('visuals/data_density.png')

suicide_data <- suicide_data %>%
    filter(year >= 1990 & year <= 2014)

suicide_data %>%
    group_by(year, country) %>%
    summarise(n_rows=n()) %>%
    filter(n_rows!=12) %>%
    nrow()

suicide_data <- suicide_data %>%
    filter(age != '5-14 years')

suicide_data %>%
    map_dbl(~ mean(is.na(.)))

suicide_data %>%
    mutate(HDI_available=!is.na(`HDI for year`)) %>%
    ggplot(aes(x=year, fill=HDI_available)) +
        geom_bar(color='black') +
        xlab('Year') + ylab('Number of cohorts') +
        ggtitle('Availability of HDI variable')

ggsave('visuals/HDI_availability.png')

suicide_data %>%
    select(suicides_no, population, `suicides/100k pop`, `gdp_for_year ($)`, `gdp_per_capita ($)`) %>%
    gather() %>%
    ggplot(aes(value)) +
        geom_histogram(bins=30, color='black', fill='white') +
        facet_wrap(key ~ ., scales='free') +
        xlab('Value') + ylab('Number of cohorts') +
        ggtitle('Distributions of numeric variables')

ggsave('visuals/numeric_variables.png')

large_countries <- suicide_data %>%
    group_by(country, year) %>%
    summarise(population = sum(population)) %>%
    group_by(country) %>%
    summarise(min_population = min(population)) %>%
    filter(min_population >= 5000000) %>%
    pull(country)

suicide_data <- suicide_data %>%
    filter(country %in% large_countries)

#### Exploratory Analysis ####

suicide_data %>%
    ggplot(aes(`suicides/100k pop`, fill=sex)) +
        geom_histogram(color='black', breaks=seq(0,200,5)) +
        scale_x_continuous(breaks=seq(0,200,20)) +
        xlab('Suicides per 100k people') + ylab('Number of cohorts') +
        ggtitle('Distribution of suicide rate')

ggsave('visuals/suicide_rate_hist.png')

suicide_data %>%
    ggplot(aes(`suicides/100k pop`, fill=sex)) +
        geom_histogram(color='black', bins=20) +
        facet_grid(sex ~ age) +
        scale_x_continuous(breaks=seq(0,200,100)) +
        xlab('Suicides per 100k people') + ylab('Number of cohorts') +
        ggtitle('Distribution of suicide rate by age')

ggsave('visuals/suicide_rate_hist_ages.png')

suicide_data %>%
    ggplot(aes(`suicides/100k pop`, fill=sex)) +
        geom_histogram(color='black', bins=20) +
        facet_grid(sex ~ generation) +
        scale_x_continuous(breaks=seq(0,200,100)) +
        xlab('Suicides per 100k people') + ylab('Number of cohorts') +
        ggtitle('Distribution of suicide rate by generation')

ggsave('visuals/suicide_rate_hist_gens.png')

suicide_data %>%
    ggplot(aes(y=country, x=`suicides/100k pop`, color=sex)) +
        geom_jitter(alpha=0.25, width=0.1, height=0.1) +
        scale_x_continuous(breaks=seq(0,200,100)) +
        xlab('Suicides per 100k people') + ylab('Country') +
        ggtitle('Cohorts by country')

ggsave('visuals/suicide_rate_countries.png')

suicide_data_australia <- suicide_data %>%
    filter(country=='Australia') %>%
    group_by(year, sex) %>%
    summarise(suicide_rate=100000*sum(suicides_no)/sum(population), country='Australia')

suicide_data_global <- suicide_data %>%
    group_by(year, sex) %>%
    summarise(suicide_rate=100000*sum(suicides_no)/sum(population), country='Other large countries')

bind_rows(suicide_data_australia, suicide_data_global) %>%
    ggplot(aes(x=year, y=suicide_rate, color=sex, alpha=country)) +
        geom_line() +
        geom_point() +
        scale_alpha_discrete(range = c(0.9, 0.35)) +
        xlab('Year') + ylab('Suicides per 100k people') +
        ggtitle('Suicide rate over time')

ggsave('visuals/rate_over_time.png')

suicide_data %>%
    filter(country=='Australia') %>%
    ggplot(aes(x=year, y=`suicides/100k pop`, color=sex)) +
        geom_line() +
        facet_wrap(~ age) +
        scale_x_continuous(breaks=seq(1990,2010,10)) +
        xlab('Year') + ylab('Suicides per 100k people') +
        ggtitle('Changes across age groups')

ggsave('visuals/rate_over_time_ages.png')

high_gdp <- c('Australia', 'Germany', 'United States')
low_gdp <- c('Guatemala', 'Uzbekistan', 'Greece')

suicide_data %>%
    filter(country %in% high_gdp | country %in% low_gdp) %>%
    group_by(country, year, sex) %>%
    summarise(suicide_rate=100000*sum(suicides_no)/sum(population), gdp_per_capita=first(`gdp_per_capita ($)`)) %>% 
    mutate(group=if_else(country %in% high_gdp, 'high_gdp', 'low_gdp')) %>%
    spread(sex, suicide_rate) %>%
    group_by(country) %>%
    arrange(country, year) %>%
    mutate(gdp_per_capita = (gdp_per_capita / first(gdp_per_capita)) - 1,
           female = (female / first(female)) - 1, 
           male = (male / first(male)) - 1) %>%
    gather(key='variable', value='value', gdp_per_capita, female, male) %>%
    mutate(variable=factor(variable, levels=c('male', 'female', 'gdp_per_capita'))) %>%
    ggplot(aes(x=year, y=value, color=variable, group=variable)) +
        geom_line(alpha=0.9) +
        facet_wrap(group ~ country) +
        scale_color_manual(values=c("#00BFC4", "#F8766D", "#999999")) +
        scale_y_continuous(labels=scales::percent_format()) +
        scale_x_continuous(breaks=seq(1990,2010,10)) +
        xlab('Year') + ylab('Percent change since 1990') +
        ggtitle('Changes in per capita GDP and suicide rates')
    
ggsave('visuals/gdp_and_suicide_rate.png')

#### Building a Model ####

model_data <- suicide_data %>%
    select(sex, age, per_capita_gdp=`gdp_per_capita ($)`, generation, 
           population=population, year, suicide_rate=`suicides/100k pop`) %>%
    filter(suicide_rate > 0.0001) %>%
    mutate(log_suicide_rate=log(suicide_rate), log_per_capita_gdp=log(per_capita_gdp),
           log_population=log(population))

model <- lm(log_suicide_rate ~ sex + age + log_per_capita_gdp + 
            generation + log_population + year + sex * log_per_capita_gdp,
            data = model_data)

summary(model)

model_summary <- autoplot(model)

ggsave('visuals/model_results.png', plot=model_summary)
