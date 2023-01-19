#!/usr/bin/env python
# coding: utf-8

# # <center>HW2: Web Scraping</center>

# <div class="alert alert-block alert-warning">Each assignment needs to be completed independently. Never ever copy others' work or let someone copy your solution (even with minor modification, e.g. changing variable names). Anti-Plagiarism software will be used to check all submissions. No last minute extension of due date. Be sure to start working on it ASAP! </div>

# ## Q1. Collecting Movie Reviews
# 
# Write a function `getReviews(url)` to scrape all **reviews on the first page**, including, 
# - **title** (see (1) in Figure)
# - **reviewer's name** (see (2) in Figure)
# - **date** (see (3) in Figure)
# - **rating** (see (4) in Figure)
# - **review content** (see (5) in Figure. For each review text, need to get the **complete text**.)
# - **helpful** (see (6) in Figure). 
# 
# 
# Requirements:
# - `Function Input`: book page URL
# - `Function Output`: save all reviews as a DataFrame of columns (`title, reviewer, rating, date, review, helpful`). For the given URL, you can get 24 reviews.
# - If a field, e.g. rating, is missing, use `None` to indicate it. 
# 
#     
# ![alt text](IMDB.png "IMDB")

# In[1]:


import requests
import pandas as pd
from bs4 import BeautifulSoup

# Add your import statements


# In[2]:



from urllib import request


def getReviews(page_url):
    reviews = None
    
    # Add your code here
    res = requests.get(page_url)
    soup = BeautifulSoup(res.text,"lxml")
    l1, l2, l3, l4, l5, l6 = [], [], [], [], [], []
    for item in soup.select(".review-container"):
        title = item.select_one(".title").get_text(strip=True)
        user = item.select_one("span.display-name-link > a").get_text(strip=True)
        date = item.select_one("span.review-date").get_text(strip=True)
        rating = item.select_one("span.rating-other-user-rating > span").get_text(strip=True)
        review = item.select_one("div.text.show-more__control").get_text()
        helpful = item.select_one("div.actions.text-muted").get_text(strip=True)
        
        l1.append(title)
        l2.append(user)
        l3.append(date)
        l4.append(rating)
        l5.append(review)
        l6.append(helpful)
    
    d = {'title':l1, 'user':l2, 'date':l3, 'rating':l4, 'review':l5, 'helpful':l6}
        
    reviews = pd.DataFrame(d)
    return reviews


# In[3]:


# Test your function

page_url = 'https://www.imdb.com/title/tt1745960/reviews?sort=totalVotes&dir=desc&ratingFilter=0'
reviews = getReviews(page_url)

print(len(reviews))
reviews.head()


# ## Q2 (Bonus) Collect Dynamic Content
# 
# Write a function `get_N_review(url, webdriver)` to scrape **at least 100 reviews** by clicking "Load More" button 5 times through Selenium WebDrive, 
# 
# 
# Requirements:
# - `Function Input`: book page `url` and a Selenium `webdriver`
# - `Function Output`: save all reviews as a DataFrame of columns (`title, reviewer, rating, date, review, helpful`). For the given URL, you can get 24 reviews.
# - If a field, e.g. rating, is missing, use `None` to indicate it. 
# 
# 

# In[67]:


def getReviews(page_url, driver):
    
    reviews = None
    driver.get(page_url)
    
    # add your code here
    
    return reviews


# In[4]:


# Test the function

executable_path = '{your web drive path}'

driver = webdriver.Firefox(executable_path=executable_path)

page_url = 'https://www.imdb.com/title/tt1745960/reviews?sort=totalVotes&dir=desc&ratingFilter=0'
reviews = getReviews(page_url, driver)

driver.quit()

print(len(reviews))
reviews.head()


# In[ ]:




