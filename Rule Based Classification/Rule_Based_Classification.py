# In this project, the goal is to create a set of rules that will define some personas and expected revenues
# we will make from a new customer belonging to that persona group. Afterwards, we will create new segments
# out of these personas to be able to give aggregated business decision on the segments and set strategies accordingly.

# Import library and read the data
import pandas as pd
from scipy.stats import shapiro, mannwhitneyu

persona = pd.read_csv("persona.csv")
df = persona.copy()
df.head()

df.describe().T  # There are no outliers
df.SOURCE.value_counts()  # There are more Android users than the IOS users.
df.SEX.value_counts()  # There are more females than the males but the difference is not that significant.
df.COUNTRY.value_counts()*100 / len(df)  # US and Brazil together represent more than 2/3 of the observations.


len(df.PRICE.value_counts())  # There are 6 different price levels. Price is more like a categorical variable
# than a numerical variable.


df.PRICE.value_counts().sort_index()  # Most of the spendings are in the mid part (29, 39, 49).

# Lets check how much revenue we made from each country, up to now.

revenue_by_country = df.groupby("COUNTRY")["PRICE"].agg("sum").sort_values(ascending=False)
countries = revenue_by_country.index
percentages = (df.COUNTRY.value_counts()*100 / len(df)).loc[list(countries)]

summary_df = pd.DataFrame({'Country': revenue_by_country.index, 'Perc. of Customers': percentages,
                                                                'Revenue': revenue_by_country})

summary_df.corr()  # Percentage of customers and revenue are perfectly correlated (0.99996). Therefore, it is safe
# to say that average spending of a customer is almost the same in all the countries.

# Average spending of a customer for each country.
df.groupby("COUNTRY")["PRICE"].agg("mean")

# Average spending of a customer for each source
df.groupby("SOURCE")["PRICE"].agg("mean")

# Average spending of a customer for each country-source pair
df.groupby(["COUNTRY", "SOURCE"])["PRICE"].agg("mean")
# Looking at this DataFrame, we can say that there might be a significant difference in the spendings
# between sources, in a specific country. (Such as, Android users in Turkey spend more than IOS users.)
# Let's test this hypothesis, using statistical tests.

turkey_android = df.loc[(df['COUNTRY'] == 'tur') & (df['SOURCE'] == 'android'), 'PRICE']
turkey_ios = df.loc[(df['COUNTRY'] == 'tur') & (df['SOURCE'] == 'ios'), 'PRICE']

shapiro(turkey_android)
shapiro(turkey_ios)

# Both the samples do not follow a normal distribution. Therefore, we will use the nonparametric ttest (mannwithenyu).

mannwhitneyu(turkey_android, turkey_ios)  # p<0.05. We can say that there is a significant difference and
# android users in Turkey spend more than ios users.

# This test can be done for each country, but it is not our area of interest as of now, therefore I will skip it.

# Let's check the average spendings w.r.t each combined level.
df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"])["PRICE"].agg("mean")

# We see that there are obvious differences between some levels. (USA IOS MALE users of age 59 spend
# 46.5 in average, well above the overall average.

# Turn these levels into indexes to start creating our rules.
agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"])["PRICE"].agg("mean").sort_values()
agg_df = agg_df.reset_index()
agg_df

# We will split the integer age values into bins, so that we can define better personas (that are not too specific).
agg_df["AGE"].describe()  # Look at the age values and its descriptive statistics.
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], [0, 19, 24, 31, 41, 70], right=False)  # Create the new bins with respect to
# my choices of bin boundaries.
agg_df["AGE_CAT"] = agg_df["AGE_CAT"].apply(lambda x: str(x.left)+"_"+str(x.right-1))  # Turn the bins into categories.
agg_df

# We are ready to go. We can create the new customer levels by combining the levels.

agg_df["customers_level_based"] = [x[0].upper()+"_"+x[1].upper()+"_"+x[2].upper()+"_"+x[5].
                                                                upper() for x in agg_df.values]

agg_df_final = agg_df[["customers_level_based", "PRICE"]].groupby("customers_level_based")["PRICE"] \
    .agg("mean").reset_index()

agg_df_final.describe()  # We have 109 personas in total. Min spending of these personas is 19 and maximum is 45.
# These values might not be reliable if the persona is created using small number of customers. However, this
# is the best we can do with the existing data. Increasing the data size will increase the reliability.

# We have a lot of personas. It is not easy to plan strategies for that many personas. Therefore, we will be
# segmenting them with respect to their average spendings.

agg_df_final["SEGMENT"] = pd.qcut(agg_df_final["PRICE"], 4, labels=["D", "C", "B", "A"])  # Create 4 segments.
agg_df_final.groupby("SEGMENT")["PRICE"].agg(["mean", "max", "sum"])  # Some statistics for each segment.

# By looking at these statistics, we can say that even though there is a difference in average spendings in
# different segments, revenue made from each segment are close. Therefore, we can't just focus on one segment
# and ignore the rest. All of them are equally important in terms of gross revenue.

# As an example, we can analyse the C segment.

agg_df_final.SEGMENT.value_counts()  # All the segments (except D) consist of 27 personas.

agg_df_final[agg_df_final["SEGMENT"] == "C"].describe()  # C segment personas spend between 32.5 and 34.1.

agg_df_final[agg_df_final["SEGMENT"] == "C"].sort_values("customers_level_based")  # Average spendings of each
# persona in the C segment.

# C segment mainly consist of females of age 19-30.

# Our classification model is ready. Now, only thing we need to do is to enter the necessary information of the
# new customer to estimate the revenue that we are going to obtain from them.

new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df_final[agg_df_final["customers_level_based"] == new_user]  # User with these features belong to A segment, and
# spends 41.83.

new_user = "FRA_IOS_FEMALE_31_40"
agg_df_final[agg_df_final["customers_level_based"] == new_user]  # User with these features belong to C segment, and
# spends 32.82.

# As an extra, I will create series of functions to automatize this work. clb_country is the main function that
# starts the computation. Note that, some personas might not exist in our data and this will result in no output
# return by the function.


def define_interval(age):
    for i in pd.cut(agg_df["AGE"], [0, 19, 24, 31, 41, 70], right=False):
        if (age >= i.left) and (age < i.right):
            return i


def clb_country():
    print("Enter Country: ")
    country = input()
    country = country[:3]
    if country.lower() in df["COUNTRY"].values:
        clb_source(country)
    else:
        print("Enter a valid country name")
        clb_country()


def clb_source(country):
    print("Enter Source: ")
    source = input()
    if source.lower() in df["SOURCE"].values:
        clb_sex(country, source)
    else:
        print("Enter a valid source")
        clb_source(country)


def clb_sex(country, source):
    print("Enter Sex: ")
    sex = input()
    if sex.lower() in df["SEX"].values:
        clb_age(country, source, sex)
    else:
        print("Enter a valid sex")
        clb_sex(country, source)


def clb_age(country, source, sex):
    print("Enter Age: ")
    try:
        age = int(input())
    except (Exception,):
        print("Enter a valid number")
        clb_age(country, source, sex)

    interval = define_interval(age)
    person = country.upper() + "_" + source.upper() + "_" + sex.upper() + "_" + str(interval.left) + "_" + \
             str(interval.right-1)
    try:
        if len(agg_df_final[agg_df_final["customers_level_based"] == person]) == 0:
            print("No such persona exists.")
        else:
            print(agg_df_final[agg_df_final["customers_level_based"] == person])
    except (Exception,):
        print("No such persona exists.")


clb_country()
