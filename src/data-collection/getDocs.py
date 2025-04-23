import pandas as pd
import time
import json
import logging
import random
import gc
import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_JSON_PATH = DATA_DIR / "scraped_documents.json"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def setup_driver():
    options = uc.ChromeOptions()
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-cache")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    options.add_argument("--disable-blink-features=AutomationControlled")
    # options.add_argument("--headless")

    driver = uc.Chrome(options=options)
    time.sleep(2)
    return driver

def scrape_articles(driver, links):
    articles_data = []

    for index, url in enumerate(links):
        try:
            driver.delete_all_cookies()
            logger.info(f"Scraping ({index + 1}/{len(links)}): {url}")
            driver.get(url)
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
            time.sleep(random.uniform(2, 5))

            soup = BeautifulSoup(driver.page_source, "html.parser")

            abstrak_header = soup.find("h2", class_="label", string="Abstrak")
            if abstrak_header:
                abstrak_paragraph = abstrak_header.find_next_sibling("p")
                if abstrak_paragraph:
                    isi_abstrak = abstrak_paragraph.get_text(strip=True)
                else:
                    isi_abstrak = "Abstrak tidak ditemukan setelah header"
            else:
                isi_abstrak = "Header Abstrak tidak ditemukan"

            article = {
                "isi": isi_abstrak
            }
            articles_data.append(article)

        except Exception as e:
            logger.warning(f"Gagal scrapping {url}: {e}")

    return articles_data

def main():
    links = [
        "https://jtiik.ub.ac.id/index.php/jtiik/article/view/4400",
        "https://jtiik.ub.ac.id/index.php/jtiik/article/view/5663",
        "https://jtiik.ub.ac.id/index.php/jtiik/article/view/4985",
        "https://jtiik.ub.ac.id/index.php/jtiik/article/view/6742",
        "https://jtiik.ub.ac.id/index.php/jtiik/article/view/3399",
    ]

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    driver = setup_driver()
    
    try:
        articles = scrape_articles(driver, links)
    finally:
        driver.quit()
        del driver
        gc.collect()


    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    logger.info(f"{len(articles)} artikel berhasil disimpan ke {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()