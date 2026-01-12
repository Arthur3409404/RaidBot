from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os

class HellhadesScraper:
    def __init__(self, driver_path=None, headless=True):
        """Initialize the Selenium WebDriver."""
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless")
        if driver_path:
            self.driver = webdriver.Chrome(executable_path=driver_path, options=options)
        else:
            self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 10)
        self.all_data = []

    def scrape_tier_list(self, url="https://hellhades.com/raid/tier-list/"):
        """Scrape the tier list data from the given URL."""
        self.driver.get(url)
        self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".tier-list-v2")))
        
        pagination_tiles = self.driver.find_elements(By.CSS_SELECTOR, ".tl-pagination .tl-pagination-tile")
        
        for tile in pagination_tiles:
            tile.click()
            time.sleep(2)  # Wait for table to load
            
            rows = self.driver.find_elements(By.CSS_SELECTOR, ".tier-list-v2 .tier-list-champion-row")
            current_champion = None
            i = 0
            while i < len(rows):
                row = rows[i]
                classes = row.get_attribute("class")
                
                if "champion-name" in classes:
                    link = row.find_element(By.TAG_NAME, "a")
                    current_champion = link.text.strip()
                    
                    rankings = []
                    for j in range(1, 6):
                        if i + j < len(rows):
                            rank_row = rows[i + j]
                            rankings.append(rank_row.text.strip())
                    
                    if current_champion and len(rankings) == 5:
                        self.all_data.append({
                            "Champion": current_champion,
                            "Overall": rankings[0],
                            "Demonlord": rankings[1],
                            "Hydra": rankings[2],
                            "Chimera": rankings[3],
                            "Amius": rankings[4],
                        })
                    i += 5  # Skip ranking rows
                i += 1

        return self.all_data

    def to_dataframe(self):
        """Convert scraped data to a pandas DataFrame."""
        return pd.DataFrame(self.all_data)

    def save_csv(self, file_path="data/database_champions/hellhades_tier_list.csv"):
        """Save the scraped data to a CSV file."""
        df = self.to_dataframe()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"Saved to {file_path}")

    def quit(self):
        """Quit the WebDriver."""
        self.driver.quit()

    def run(self):
        try:
            self.scrape_tier_list()
            self.save_csv("data/database_champions/hellhades_tier_list.csv")
        finally:
            self.quit()
