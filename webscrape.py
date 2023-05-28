from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import io
import pandas as pd
import numpy as np
import csv


def scrapfyt(url):

    # Opening chrome and url
    option = webdriver.ChromeOptions()

    option.add_argument('--headless')
    option.add_argument('-no-sandbox')
    option.add_argument("--disable-infobars")
    option.add_argument("--disable-gpu")
    option.add_argument("--mute-audio")
    option.add_argument("--disable-extensions")
    option.add_argument('-disable-dev-shm-usage')

    driver = webdriver.Chrome(service=Service(
        r"D:/YT Sent Analysis/chromedriver.exe"), options=option)

    driver.set_window_size(960, 800)
    """
    # minimizing window to optimum because of youtube design of
    # right side videos recommendations. When in max window,
    # while scrolling comments, it cannot be able to load correctly
    # due to the video recommendations on the right side.
    """

    time.sleep(1)
    driver.get(url)
    time.sleep(2)

    # Scrolling
    driver.execute_script("window.scrollBy(0,500)")

    last_height = driver.execute_script(
        "return document.documentElement.scrollHeight")

    while True:
        # Scroll down until "next load".
        driver.execute_script(
            "window.scrollTo(0, document.documentElement.scrollHeight);")

        # Wait to load everything thus far.
        time.sleep(4)

        # Calculate new scroll height and compare with last scroll height.
        new_height = driver.execute_script(
            "return document.documentElement.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    driver.execute_script(
        "window.scrollTo(0, document.documentElement.scrollHeight);")

    # Change max-number-of-lines attribute to 40 for all ytd-expander tags
    expander_tags = driver.find_elements(By.TAG_NAME, 'ytd-expander')
    for expander_tag in expander_tags:
        driver.execute_script(
            "arguments[0].setAttribute('max-number-of-lines', '40');", expander_tag)

    # Scraping all the comments
    users = driver.find_elements(By.XPATH, '//*[@id="author-text"]/span')
    comments = driver.find_elements(By.XPATH, '//*[@id="content-text"]')

    with io.open('comments.csv', 'w', newline='', encoding="utf-16") as file:
        writer = csv.writer(file, delimiter=",", quoting=csv.QUOTE_ALL)
        writer.writerow(["Username", "Comment"])
        for username, comment in zip(users, comments):
            writer.writerow([username.text, comment.text])

    commentsfile = pd.read_csv("comments.csv", encoding="utf-16")

    all_comments = commentsfile.replace(np.nan, '-', regex=True)
    all_comments = all_comments.to_csv("Full Comments.csv", index=False)

    video_comment_without_replies = str(
        len(commentsfile.axes[0])) + ' Comments'
    driver.close()
    return video_comment_without_replies


link = input("Enter URL: ")
s = scrapfyt(link)
print(s)
