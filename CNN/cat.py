# https://aiacademy.jp/media/?p=352
# from icrawler.builtin import GoogleImageCrawler
# 猫の画像を100枚取得
# crawler = GoogleImageCrawler(storage={"root_dir": "cats"})
# crawler.crawl(keyword="猫", max_num=100)
from icrawler.builtin import BingImageCrawler

# 猫の画像を100枚取得
crawler = BingImageCrawler(storage={"root_dir": "cat"})
crawler.crawl(keyword="猫", max_num=100)