from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class LinkGenerator(object):

    driver = None

    def __init__(self, category_link):

        if self.driver is None:
            self.driver = webdriver.Chrome()
        self.category_link = category_link
        self.driver.get(category_link)
        self.pages_df = []

    def search_for_links(self):

        user_data = self.driver.find_elements_by_xpath('//*[@id="video-title"]')
        links = []
        for i in user_data:
            links.append(i.get_attribute('href'))
            if len(links) > 10000:
                break
        new_page_df = pd.DataFrame({'link': links, 'category': [self.category_link] * len(links)})
        self.pages_df.append(new_page_df)
        print(len(links))

    def __compress_pages_df(self):

        self.pages_df = [pd.concat(tuple(self.pages_df))]

    def persist_data(self):
        for df in self.pages_df:
            df.to_csv('result.csv', 'a+')

if __name__ == '__main__':

    new_generator = LinkGenerator('https://www.youtube.com/results?search_query=toy+unbox&sp=EgIQAQ%253D%253D')
    new_generator.search_for_links()
    new_generator.persist_data()
