from utils.driver_setup import create_driver
from crawler.dmxcrawler import crawl_dmx_reviews
from storage.csv_storage import save_to_csv
from config.settings import OUTPUT_FILE

def main():

    driver = create_driver()

    all_reviews = []

    try:

        while True:

            print(
                """
========================================
Mở sản phẩm cần crawl trên trình duyệt.

- Nhấn Enter để crawl sản phẩm hiện tại
- Nhập 0 để dừng
========================================
"""
            )

            choice = input(">>> Lựa chọn: ").strip()

            if choice == "0":
                break

            # crawl sản phẩm đang mở
            reviews = crawl_dmx_reviews(driver)

            # cộng vào danh sách tổng
            all_reviews.extend(reviews)

            print(
                f"\nĐã crawl sản phẩm này: {len(reviews)} review"
            )

            print(
                f"Tổng review hiện tại: {len(all_reviews)}"
            )

        # ========================================
        # SAU KHI USER NHẬP 0
        # ========================================

        save_to_csv(
            all_reviews,
            OUTPUT_FILE
        )

    finally:

        driver.quit()


if __name__ == "__main__":
    main()