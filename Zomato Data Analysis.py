#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('zomato.csv',encoding='latin-1')
df.head()


# ## Checking if dataset contains any null

# In[4]:


## Checking if dataset contains any null

nan_values = df.isna()
nan_columns = nan_values.any()

columns_with_nan = df.columns[nan_columns].tolist()
print(columns_with_nan)


#  Cuisines seems to contain null values. Hence any further analysis involving Cuisines the NaN values has to be considered.
#  There is an other file which is also available along with this dataset is a Country Code.

# In[5]:


df1 = pd.read_excel('Country-Code.xlsx')
df1.head()


# Let us merge both the datasets. This will help us to understand the dataset country wise.

# In[6]:


df2 = pd.merge(df,df1,on='Country Code',how='left')
df2.head(2)


# ## Exploratory Analysis and Visualization
# Before we ask question on the dataset, it would be helpful to understand the restaurants geographical spread, understanding the rating, Currency, Online Delivery, City coverage…etc.
# ### List of countries the survey is spread across

# In[7]:


print('List of counteris the survey is spread accross - ')
for x in pd.unique(df2.Country): print(x)
print()
print('Total number to country', len(pd.unique(df2.Country)))


# ####  The survey seems to have spread across15 countries. This shows that Zomato is a multinational company having actives business in all those countries.

# In[8]:


from plotly.offline import init_notebook_mode, plot, iplot

labels = list(df2.Country.value_counts().index)
values = list(df2.Country.value_counts().values)

fig = {
    "data":[
        {
            "labels" : labels,
            "values" : values,
            "hoverinfo" : 'label+percent',
            "domain": {"x": [0, .9]},
            "hole" : 0.6,
            "type" : "pie",
            "rotation":120,
        },
    ],
    "layout": {
        "title" : "Zomato's Presence around the World",
        "annotations": [
            {
                "font": {"size":20},
                "showarrow": True,
                "text": "Countries",
                "x":0.2,
                "y":0.9,
            },
        ]
    }
}

iplot(fig)


# ### As Zomato is a startup from India hence it makes sense that it has maximum business spread across restaurants in India

# ## Understanding the Rating aggregate, color and text

# In[9]:


df3 = df2.groupby(['Aggregate rating','Rating color', 'Rating text']).size().reset_index().rename(columns={0:'Rating Count'})
df3
df3


# The above information helps us to understand the relation between Aggregate rating, color and text. We conclude the following color assigned to the ratings:
# Rating 0 — White — Not rated
# Rating 1.8 to 2.4 — Red — Poor
# Rating 2.5 to 3.4 — Orange — Average
# Rating 3.5 to 3.9 — Yellow — Good
# Rating 4.0 to 4.4 — Green — Very Good
# Rating 4.5 to 4.9 — Dark Green — Excellent
# 
# Let us try to understand the spread of rating across restaurants

# In[10]:


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

plt.figure(figsize=(12,6))
# plt.xticks(rotation=75)
plt.title('Rating Color')
sns.barplot(x=df3['Rating color'], y=df3['Rating Count']);


# Interesting, Maximum restaurants seems to have gone No ratings. Let us check if these restaurants belong to some specific country.

# In[11]:


No_rating = df2[df2['Rating color']=='White'].groupby('Country').size().reset_index().rename(columns={0:'Rating Count'})
No_rating


# India seems to have maximum unrated restaurants. In India the culture of ordering online food is still gaining momentum hence most of the restaurants are still unrated on Zomato as people might be preferring to visiting the restaurant for a meal.

# ## Country and Currency

# In[12]:


country_currency = df2[['Country','Currency']].groupby(['Country','Currency']).size().reset_index(name='count').drop('count', axis=1, inplace=False)
country_currency.sort_values('Currency').reset_index(drop=True)


# Above table display country and the currency they accept. Interestingly four countries seems to be accepting currency in dollars.

# ## Online delivery distribution

# In[13]:


plt.figure(figsize=(12,6))
plt.title('Online Delivery Distribution')
plt.pie(df2['Has Online delivery'].value_counts()/9551*100, labels=df2['Has Online delivery'].value_counts().index, autopct='%1.2f%%', startangle=180);


# Only 25% of restaurants accepts online delivery. This data might be biased as we have maximum restaurants listed here are from India. Maybe analysis over city wise would be more helpful.

# ## Let us try to understand the coverage of city

# In[14]:


from plotly.offline import init_notebook_mode, plot, iplot
import plotly.graph_objs as go
plt.figure(figsize=(12,6))
# import plotly.plotly as py

labels = list(df2.City.value_counts().head(20).index)
values = list(df2.City.value_counts().head(20).values)

fig = {
    "data":[
        {
            "labels" : labels,
            "values" : values,
            "hoverinfo" : 'label+percent',
            "domain": {"x": [0, .9]},
            "hole" : 0.6,
            "type" : "pie",
            "rotation":120,
        },
    ],
    "layout": {
        "title" : "Zomato's Presence Citywise",
        "annotations": [
            {
                "font": {"size":20},
                "showarrow": True,
                "text": "Cities",
                "x":0.2,
                "y":0.9,
            },
        ]
    }
}
iplot(fig);


# The data seems to be skewed towards New Delhi, Gurgaon and Noida. I see minimal data for other cities. Hence I would do my analysis predominantly on New Delhi.
# 
# ## Asking and Answering Questions
# 
# We’ve already gained several insights about the restaurants present in the survey. Let’s ask some specific questions and try to answer them using data frame operations and visualizations.

# # Q1: From which Locality maximum hotels are listed in Zomato

# In[15]:


Delhi = df2[(df2.City == 'New Delhi')]
plt.figure(figsize=(12,6))
sns.barplot(x=Delhi.Locality.value_counts().head(10), y=Delhi.Locality.value_counts().head(10).index)

plt.ylabel(None);
plt.xlabel('Number of Resturants')
plt.title('Resturants Listing on Zomato');


# Connaught place seems to have high no of restaurants registered with Zomato, Let us understand the cuisines the top rated restaurants has to offer

# ## Q2: What kind of Cuisine these highly rates restaurants offer

# In[16]:


# I achieve this by the following steps

## Fetching the resturants having 'Excellent' and 'Very Good' rating
ConnaughtPlace = Delhi[(Delhi.Locality.isin(['Connaught Place'])) & (Delhi['Rating text'].isin(['Excellent','Very Good']))]

ConnaughtPlace = ConnaughtPlace.Cuisines.value_counts().reset_index()

## Extracing all the cuisens in a single list
cuisien = []
for x in ConnaughtPlace['index']: 
  cuisien.append(x)

# cuisien = '[%s]'%', '.join(map(str, cuisien))
cuisien


# In[17]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
  
comment_words = ''
stopwords = set(STOPWORDS)
  
# iterate through the csv file
for val in cuisien:
      
    # typecaste each val to string
    val = str(val)
  
    # split the value
    tokens = val.split()
      
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
      
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 1500, height = 1500,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
  
# plot the WordCloud image                       
plt.figure(figsize = (8, 8), facecolor = 'orange')
plt.title('Resturants cuisien -  Top Resturants')
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()


# Top rated restaurants seems to be doing well in the following cuisine
# * North Indian
# * Chinese
# * Italian
# * American

# ## Q3: How many of such restaurants accept online delivery

# In[18]:


top_locality = Delhi.Locality.value_counts().head(10)
sns.set_theme(style="darkgrid")
plt.figure(figsize=(12,6))
ax = sns.countplot(y= "Locality", hue="Has Online delivery", data=Delhi[Delhi.Locality.isin(top_locality.index)])
plt.title('Resturants Online Delivery');


# * Apart from Shahdara locality, restaurants in other locality accepts online delivery.
# * Online Delivery seems to be on higher side in Defence colony and Malviya Nagar

# ## Q4: Understanding the Restaurants Rating localities.

# * Apart from Malviya nagar, Defence colony in rest of the locality people seems to prefer visiting the restaurants rather ordering food online.
# * I would now like to understand the rating of these restaurants that are providing online delivery in Malviya nagar, Defence colony.

# ![image.png](attachment:image.png)

# * Defence colony seems to have high no of highly rated restaurants but Malviya Nagar seems to done better in terms of Good and Average restaurants.
# * As restaurants with ‘Poor’ and ‘Not Rated’ is far lesser that ‘Good’, ‘Very Good’ and ‘Excellent’ restaurants. Hence people in these localities prefer online ordering

# ## Q5: Rating VS Cost of dinning

# In[18]:


plt.figure(figsize=(12,6))
sns.scatterplot(x="Average Cost for two", y="Aggregate rating", hue='Price range', data=Delhi)

plt.xlabel("Average Cost for two")
plt.ylabel("Aggregate rating")
plt.title('Rating vs Cost of Two');


# I observe there is no linear relation between price and rating. For instance, Restaurants with good rating (like 4–5) have restaurants with all the price range and spread across the entire X axis

# ## Q6: Location of Highly rated restaurants across New Delhi

# In[19]:


Delhi['Rating text'].value_counts()


# In[20]:


import plotly.express as px
Highly_rated = Delhi[Delhi['Rating text'].isin(['Excellent'])]

fig = px.scatter_mapbox(Highly_rated, lat="Latitude", lon="Longitude", hover_name="City", hover_data=["Aggregate rating", "Restaurant Name"],
                        color_discrete_sequence=["fuchsia"], zoom=10, height=300)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_layout(title='Highle rated Resturants Location',
                  autosize=True,
                  hovermode='closest',
                  showlegend=False)
fig.update_layout(
    autosize=False,
    width=800,
    height=500,)

fig.show()


# The aforementioned four cities represent nearly 65% of the total data available in the dataset. Apart from the higly rated local restaurants, it’d be intersting to know where the known-eateries that are commonplace. The verticles across which these can be located are -
# * Breakfast
# * American Fast Food
# * Ice Creams, Shakes & Desserts

# ## Q7: Common Eateries

# ###  1: Breakfast and Coffee locations

# In[21]:


types = {
    "Breakfast and Coffee" : ["Cafe Coffee Day", "Starbucks", "Barista", "Costa Coffee", "Chaayos", "Dunkin' Donuts"],
    "American": ["Domino's Pizza", "McDonald's", "Burger King", "Subway", "Dunkin' Donuts", "Pizza Hut"],
    "Ice Creams and Shakes": ["Keventers", "Giani", "Giani's", "Starbucks", "Baskin Robbins", "Nirula's Ice Cream"]
}

breakfast = Delhi[Delhi['Restaurant Name'].isin(types['Breakfast and Coffee'])]
american = Delhi[Delhi['Restaurant Name'].isin(types['American'])]
ice_cream = Delhi[Delhi['Restaurant Name'].isin(types['Ice Creams and Shakes'])]


# In[22]:


breakfast = breakfast[['Restaurant Name','Aggregate rating']].groupby('Restaurant Name').mean().reset_index().sort_values('Aggregate rating',ascending=False)
breakfast


# In[23]:


import plotly.express as px

df= breakfast
fig = px.bar(df, y='Aggregate rating', x='Restaurant Name', text='Aggregate rating', title="Breakfast and Coffee locations")
fig.update_traces(texttemplate='%{text:.3s}', textposition='outside')
fig.update_layout(
    autosize=False,
    width=800,
    height=500,)
fig.show()


# Chaayos outlets are doing better. We need more of those in Delhi. Café coffee day seems to be performing poorly in avg rating. They are required to improve their services.

# ### 2: Fast Food Restaurants

# In[24]:


american = american[['Restaurant Name','Aggregate rating']].groupby('Restaurant Name').mean().reset_index().sort_values('Aggregate rating',ascending=False)
american


# In[25]:


import plotly.express as px

df= american
fig = px.bar(df, y='Aggregate rating', x='Restaurant Name', text='Aggregate rating', title="Fast Food Resturants")
fig.update_traces(texttemplate='%{text:.3s}', textposition='outside')
fig.update_layout(
    autosize=False,
    width=800,
    height=500,)

fig.show()


# ### 3: Ice Cream Parlors

# In[26]:


ice_cream = ice_cream[['Restaurant Name','Aggregate rating']].groupby('Restaurant Name').mean().reset_index().sort_values('Aggregate rating',ascending=False)
ice_cream


# In[27]:


import plotly.express as px

df= ice_cream
fig = px.bar(df, y='Aggregate rating', x='Restaurant Name', text='Aggregate rating', title="Ice Cream Parlours")
fig.update_traces(texttemplate='%{text:.3s}', textposition='outside')
fig.update_layout(
    autosize=False,
    width=800,
    height=500,)
fig.show()


# Foreign brands seems to be doing better than the local brands

# ## Inferences and Conclusions

# We’ve drawn many inferences from the survey. Here’s a summary of a few of them:
# * The dataset is skewed towards India and doesn't represent the complete data of restaurants worldwide.
# * Restaurants rating is categorized in categories
#    * Not Rated
#    * Average
#    * Good
#    * Very Good
#    * Excellent
# * Connaught Palace have maximum restaurants listed on Zomato but in terms of online delivery acceptance Defence colony and Malviya nagar seems to be doing better.
# * The top rated restaurants seems to be getting better rating on the following cuisine
#     * North Indian
#     * Chinese
#     * American
#     * Italian
# * There is no relation between cost and rating. Some of the best rated restaurants are low on cost and vice versa.
# * On common Eateries, For Breakfast and Coffee location Indian restaurants seems to be better rated but for Fast food chain and Ice cream parlors American restaurants seems to be doing better.

# In[ ]:




