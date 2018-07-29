library(tidyverse)
library(readxl)
library(ggpubr)

#### Import and clean data ####

datafile <- "./data/speed-dating-data.csv"

raw_data <- read_csv(datafile) %>%
  select(1:122)

individual_data <- raw_data %>%
  group_by(i_id) %>%
  summarise(
            # Basic information
            w_id = first(w_id),
            # Individual information
            Gender = first(gender), Age = first(age), `Field of Study` = first(field), `Field Code` = first(field_cd), Undergrad = first(undergra),
            `Median SAT` = first(mn_sat), `Tution Fees` = parse_number(first(tuition)), `Race Code` = first(race), `Importance of Race` = first(imprace),
            `Importance of Religion` = first(imprelig), Origin = first(from), Zipcode = first(zipcode), `Median Origin Income` = parse_number(first(income)),
            `Goal Code` = first(goal), `Date Frequency` = first(date), `Out Frequency` = first(go_out),
            Career = first(career), `Career Code` = first(career_c),
            # Individual interests
            int_Sports = first(sports), int_TVSports = first(tvsports), int_Exercise = first(exercise), int_DiningOut = first(dining), 
            int_Museums = first(museums), int_Art = first(art), int_Hiking = first(hiking), int_Gaming = first(gaming), int_Clubbing = first(clubbing),
            int_Reading = first(reading), int_TV = first(tv), int_Theatre = first(theater), int_Movies = first(movies), int_Concerts = first(concerts),
            int_Music = first(music), int_Shopping = first(shopping), int_Yoga = first(yoga),
            # Stated preferences
            pref_Attractiveness = first(attr1_1), pref_Sincerity = first(sinc1_1), pref_Intelligence = first(intel1_1), 
            pref_Fun = first(fun1_1), pref_Ambition = first(amb1_1), pref_SharedInterests = first(shar1_1),
            # Thoughts on opposite sex
            opp_Attractiveness = first(attr2_1), opp_Sincerity = first(sinc2_1), opp_Intelligence = first(intel2_1), opp_Fun= first(fun2_1),
            opp_Ambition = first(amb2_1), opp_SharedInterests = first(shar2_1),
            # Evaluations by partners
            actual_Attractiveness = mean(attr_o, na.rm = TRUE), actual_Sincerity = mean(sinc_o, na.rm = TRUE), actual_Intelligence = mean(intel_o, na.rm = TRUE), actual_Fun = mean(fun_o, na.rm = TRUE),
            actual_Ambition = mean(amb_o, na.rm = TRUE),
            # Self-evaluation
            self_Attractiveness = first(attr3_1), self_Sincerity = first(sinc3_1), self_Intelligence = first(intel3_1), self_Fun= first(fun3_1),
            self_Ambition = first(amb3_1)
            ) %>%
  mutate(Gender = as_factor(if_else(Gender == 0, "Female", "Male")))

N_individuals <- dim(individual_data)[1]
N_female <- sum(individual_data$Gender == "Female")
N_male <- sum(individual_data$Gender == "Male")

#### Analysis of interests ####

individual_interests <- individual_data %>%
  select(Gender, starts_with("int_")) %>%
  gather(2:18, key="Interest", value="Score") %>%
  mutate(Interest = str_sub(Interest, 5)) %>%
  filter(Score <= 10)

ggplot(data = individual_interests) +
  geom_bar(aes(x =Interest, y=Score, fill=Gender ), position='dodge', stat='summary', fun.y='mean') +
  ggtitle("Stated Interests", subtitle = "By Gender") +
  ylab('Average Score')

ggsave("./output/interests_bygender.png", width = 15, height = 6)

#### Analysis of stated preferences ####

# Simple comparisons

individual_preferences <- individual_data %>%
  select(Gender, starts_with("pref_")) %>%
  gather(2:7, key="Attribute", value="Scoring") %>%
  mutate(Attribute = str_sub(Attribute, start = 6))

ggplot(data = individual_preferences) +
  stat_boxplot(mapping = aes( x = Attribute, y = Scoring)) +
  ggtitle("Individual Stated Preferences")

ggsave("./output/preferences_overall.png", width = 9, height = 5)
  
ggplot(data = individual_preferences) +
  stat_boxplot(mapping = aes( x = Attribute, y = Scoring, fill = Gender)) +
  ggtitle("Individual Stated Preferences", subtitle = "By Gender")

ggsave("./output/preferences_bygender.png", width = 9, height = 5)

individual_preferences <- individual_data %>%
  select(Gender, starts_with("opp_")) %>%
  gather(2:7, key="Attribute", value="Scoring") %>%
  mutate(Attribute = str_sub(Attribute, start = 5))

ggplot(data = individual_preferences) +
  stat_boxplot(mapping = aes( x = Attribute, y = Scoring, fill = Gender)) +
  ggtitle("Perceived Preferences", subtitle = "Of Opposite Sex") +
  scale_fill_discrete(labels = c("Women think men want...", "Men think women want..."))

ggsave("./output/preferences_opp_sex.png", width = 9, height = 5)

#### Analysis of actuals ####

individuals_by_attributes <- individual_data %>%
  select(i_id, Gender, starts_with("actual_"), starts_with("self_"))

individuals_each_score <- individuals_by_attributes %>%
  gather("Attribute", "Score", 3:12) %>%
  mutate(Level = str_extract(Attribute, pattern = ".+(?=_)"), Attribute = str_extract(Attribute, pattern = "(?<=_).+")) %>%
  spread(Level, Score)

individual_overall_scores <- individuals_by_attributes %>%
  mutate(Actual = actual_Ambition + actual_Attractiveness + actual_Fun + actual_Intelligence + actual_Sincerity,
         Self = self_Ambition + self_Attractiveness + self_Fun + self_Intelligence + self_Sincerity)

# Look at actuals for each category by gender

plt1 <- ggplot(data = filter(individuals_each_score, Attribute == "Ambition")) +
  geom_density(mapping = aes(x = actual, fill = Gender)) +
  theme(axis.text.y=element_blank(), 
        axis.ticks=element_blank(), 
        axis.title.y=element_blank()) +
  ggtitle("Ambition") +
  xlab('Actual Score')

plt2 <- ggplot(data = filter(individuals_each_score, Attribute == "Attractiveness")) +
  geom_density(mapping = aes(x = actual, fill = Gender)) +
  theme(axis.text.y=element_blank(), 
        axis.ticks=element_blank(), 
        axis.title.y=element_blank()) +
  ggtitle("Attractiveness") +
  xlab('Average Score')

plt3 <- ggplot(data = filter(individuals_each_score, Attribute == "Fun")) +
  geom_density(mapping = aes(x = actual, fill = Gender)) +
  theme(axis.text.y=element_blank(), 
        axis.ticks=element_blank(), 
        axis.title.y=element_blank()) +
  ggtitle("Fun") +
  xlab('Average Score')

plt4 <- ggplot(data = filter(individuals_each_score, Attribute == "Intelligence")) +
  geom_density(mapping = aes(x = actual, fill = Gender)) +
  theme(axis.text.y=element_blank(), 
        axis.ticks=element_blank(), 
        axis.title.y=element_blank()) +
  ggtitle("Intelligence") +
  xlab('Average Score')

plt5 <- ggplot(data = filter(individuals_each_score, Attribute == "Sincerity")) +
  geom_density(mapping = aes(x = actual, fill = Gender)) +
  theme(axis.text.y=element_blank(), 
        axis.ticks=element_blank(), 
        axis.title.y=element_blank()) +
  ggtitle("Sincerity") +
  xlab('Average Score')

plt6 <- ggplot(data = individual_overall_scores) +
  geom_density(mapping = aes(x = Actual , fill = Gender)) +
  theme(axis.text.y=element_blank(), 
        axis.ticks=element_blank(), 
        axis.title.y=element_blank()) +
  ggtitle("Overall") +
  xlab('Average Score')

fig <- ggarrange(plt1, plt2, plt3, plt4, plt5, plt6, ncol = 2, nrow = 3, common.legend = TRUE)
annotate_figure(fig, top = text_grob("A Battle of the Sexes", face = "bold", size = 22))

ggsave("./output/scores_distributions_actual.png", width = 12, height = 8)

rm(fig, plt1, plt2, plt3, plt4, plt5, plt6)

# Look at self-assessments for each category by gender

plt1 <- ggplot(data = filter(individuals_each_score, Attribute == "Ambition")) +
  geom_density(mapping = aes(x = self, fill = Gender, alpha = 0.6)) +
  ggtitle("Ambition")

plt2 <- ggplot(data = filter(individuals_each_score, Attribute == "Attractiveness")) +
  geom_density(mapping = aes(x = self, fill = Gender, alpha = 0.6)) +
  ggtitle("Attractiveness")

plt3 <- ggplot(data = filter(individuals_each_score, Attribute == "Fun")) +
  geom_density(mapping = aes(x = self, fill = Gender, alpha = 0.6)) +
  ggtitle("Fun")

plt4 <- ggplot(data = filter(individuals_each_score, Attribute == "Intelligence")) +
  geom_density(mapping = aes(x = self, fill = Gender, alpha = 0.6)) +
  ggtitle("Intelligence")

plt5 <- ggplot(data = filter(individuals_each_score, Attribute == "Sincerity")) +
  geom_density(mapping = aes(x = self, fill = Gender, alpha = 0.6)) +
  ggtitle("Sincerity")

plt6 <- ggplot(data = individual_overall_scores) +
  geom_density(mapping = aes(x = Self , fill = Gender, alpha = 0.6)) +
  ggtitle("Overall")

fig <- ggarrange(plt1, plt2, plt3, plt4, plt5, plt6, ncol = 2, nrow = 3, common.legend = TRUE)
annotate_figure(fig, top = text_grob("A Battle of the Sexes: Self", face = "bold", size = 22))

ggsave("./output/scores_distributions_self.png", width = 12, height = 8)

rm(fig, plt1, plt2, plt3, plt4, plt5, plt6)

# Compare the two

plt1 <- ggplot(data = filter(individuals_each_score, Attribute == "Ambition")) +
  geom_jitter(mapping = aes(x = self, y = actual, color = Gender), width = 0.1) +
  geom_abline(slope = 1, intercept = 0) +
  ggtitle("Ambition") +
  xlab('Self-assessment') +
  ylab('Average Score')

plt2 <- ggplot(data = filter(individuals_each_score, Attribute == "Attractiveness")) +
  geom_jitter(mapping = aes(x = self, y = actual, color = Gender), width = 0.1) +
  geom_abline(slope = 1, intercept = 0) +
  ggtitle("Attractiveness") +
  xlab('Self-assessment') +
  ylab('Average Score')

plt3 <- ggplot(data = filter(individuals_each_score, Attribute == "Fun")) +
  geom_jitter(mapping = aes(x = self, y = actual, color = Gender), width = 0.1) +
  geom_abline(slope = 1, intercept = 0) +
  ggtitle("Fun") +
  xlab('Self-assessment') +
  ylab('Average Score')

plt4 <- ggplot(data = filter(individuals_each_score, Attribute == "Intelligence")) +
  geom_jitter(mapping = aes(x = self, y = actual, color = Gender), width = 0.1) +
  geom_abline(slope = 1, intercept = 0) +
  ggtitle("Intelligence") +
  xlab('Self-assessment') +
  ylab('Average Score')

plt5 <- ggplot(data = filter(individuals_each_score, Attribute == "Sincerity")) +
  geom_jitter(mapping = aes(x = self, y = actual, color = Gender), width = 0.1) +
  geom_abline(slope = 1, intercept = 0) +
  ggtitle("Sincerity") +
  xlab('Self-assessment') +
  ylab('Average Score')

plt6 <- ggplot(data = individual_overall_scores) +
  geom_jitter(mapping = aes(x = Self, y = Actual, color = Gender)) +
  geom_abline(slope = 1, intercept = 0) +
  ggtitle("Overall") +
  xlab('Self-assessment') +
  ylab('Average Score')

fig <- ggarrange(plt1, plt2, plt3, plt4, plt5, plt6, ncol = 2, nrow = 3, common.legend = TRUE)
annotate_figure(fig, top = text_grob("How Do We Measure Up?", face = "bold", size = 22))

ggsave("./output/scores_comparisons.png", width = 12, height = 8)

rm(fig, plt1, plt2, plt3, plt4, plt5, plt6)

# See who is above and below the line

summary <- individuals_each_score %>%
  group_by(Gender, Attribute) %>%
  summarise('Below Line' = mean(if_else(actual<self,1,0), na.rm = TRUE)) %>%
  arrange(Attribute) %>%
  spread(key=Gender, value=`Below Line`)

#### Modelling matches ####

# Generate match data

partner_decisions <- raw_data %>%
  select(i_id, p_i_id, dec_o) %>%
  rename(individual_id = i_id, partner_id = p_i_id)

match_details <- raw_data %>%
  select(i_id, gender, order, init_corr, samerace, age,
         match, 
         p_i_id, age_o, ends_with("1_1")) %>%
  rename(pref_Attractiveness = attr1_1, pref_Sincerity = sinc1_1, pref_Intelligence = intel1_1, 
          pref_Fun = fun1_1, pref_Ambition = amb1_1, pref_SharedInterests = shar1_1,
          i_Gender = gender) %>%
  left_join(individuals_by_attributes, by = c("p_i_id" = "i_id")) %>%
  select(-Gender, -starts_with("self_")) %>%
  rename(p_age = age_o, i_age = age) %>%
  left_join(partner_decisions, by = c("i_id" = "partner_id", "p_i_id" = "individual_id")) %>%
  rename(wants_to_match = dec_o)

# Take a simple initial model for desire to match

X <- match_details %>%
  mutate(age_diff = i_age - p_age) %>%
  select(-c(i_age, p_age, i_id, p_i_id, match, starts_with("pref"))) %>%
  na.omit()

X <- slice(X, sample(nrow(X)))

X_train <- X[1:floor(0.8*nrow(X)),]
X_test <- X[floor(0.8*nrow(X)):nrow(X),]

model <- glm(wants_to_match ~ ., family = binomial(link = 'logit'), data = X_train)

summary(model)

results <- predict(model, newdata = X_test)
predictions <- ifelse(results > 0.5, 1, 0)
accuracy <- mean(predictions == X_test$wants_to_match)
print(accuracy)

# Take a more complex model incorporating the differences between stated and actual preferences

X <- match_details %>%
  mutate(age_diff = i_age - p_age) %>%
  select(-c(i_age, p_age, i_id, p_i_id, match)) %>%
  na.omit() %>%
  mutate(
    d_Attractiveness = (actual_Attractiveness - mean(actual_Attractiveness))/sd(actual_Attractiveness) - (pref_Attractiveness - mean(pref_Attractiveness))/sd(pref_Attractiveness),
    d_Sincerity = (actual_Sincerity - mean(actual_Sincerity))/sd(actual_Sincerity) - (pref_Sincerity - mean(pref_Sincerity))/sd(pref_Sincerity),
    d_Intelligence = (actual_Intelligence - mean(actual_Intelligence))/sd(actual_Intelligence) - (pref_Intelligence - mean(pref_Intelligence))/sd(pref_Intelligence),
    d_Fun = (actual_Fun - mean(actual_Fun))/sd(actual_Fun) - (pref_Fun - mean(pref_Fun))/sd(pref_Fun),
    d_Ambition = (actual_Ambition - mean(actual_Ambition))/sd(actual_Ambition) - (pref_Ambition - mean(pref_Ambition))/sd(pref_Ambition)
  ) %>%
  select(-c(starts_with('actual_'), starts_with('pref_')))

X <- slice(X, sample(nrow(X)))

X_train <- X[1:floor(0.8*nrow(X)),]
X_test <- X[floor(0.8*nrow(X)):nrow(X),]

model <- glm(wants_to_match ~ ., family = binomial(link = 'logit'), data = X_train)

summary(model)

results <- predict(model, newdata = X_test)
predictions <- ifelse(results > 0.5, 1, 0)
accuracy <- mean(predictions == X_test$wants_to_match)
print(accuracy)
