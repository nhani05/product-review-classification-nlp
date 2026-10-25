"""
Crawler module for collecting customer reviews from Dien May Xanh (dienmayxanh.com).
Refactored and organized from crawldatadmx into a modular script supporting CLI usage.
"""

import os
import time
import argparse
from typing import List, Optional, Dict, Any


class Review:
    """Data model representing a single crawled review."""

    def __init__(
        self,
        text: str,
        rating: Optional[int] = None,
        username: Optional[str] = None,
        product_name: Optional[str] = None,
        source_url: Optional[str] = None,
    ):
        self.text = (text or "").strip()
        self.rating = rating
        self.username = (username or "").strip() or None
        self.product_name = (product_name or "").strip() or None
        self.source_url = (source_url or "").strip() or None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "noi_dung": self.text,
            "so_sao": self.rating,
            "nguoi_danh_gia": self.username,
            "san_pham": self.product_name,
            "link": self.source_url,
        }


def create_driver(headless: bool = False):
    """
    Initializes a Chrome Selenium WebDriver with anti-detection options.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError as exc:
        raise ImportError(
            "Missing Selenium or webdriver-manager dependencies. "
            "Please run: pip install -r requirements.txt"
        ) from exc

    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    if headless:
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    if not headless:
        driver.maximize_window()
    return driver


def load_more_reviews(driver, max_clicks: int = 50) -> None:
    """Repeatedly clicks the 'Xem thêm' button up to max_clicks times."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.common.exceptions import (
        StaleElementReferenceException,
        WebDriverException,
    )

    print("\n[+] Đang tải thêm đánh giá...")
    for click_idx in range(max_clicks):
        xpaths = [
            "//button[contains(normalize-space(.), 'Xem thêm')]",
            "//a[contains(normalize-space(.), 'Xem thêm')]",
            "//*[self::button or self::a][contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'xem thêm')]",
        ]
        candidates = []
        seen_ids = set()

        for xpath in xpaths:
            try:
                for element in driver.find_elements(By.XPATH, xpath):
                    try:
                        elem_id = element.id
                        if elem_id not in seen_ids:
                            seen_ids.add(elem_id)
                            candidates.append(element)
                    except Exception:
                        continue
            except Exception:
                continue

        clicked = False
        for button in candidates:
            try:
                if not button.is_displayed() or not button.is_enabled():
                    continue

                label = button.text.strip().casefold()
                if "xem thêm" not in label:
                    continue

                driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center'});", button
                )
                time.sleep(0.4)

                try:
                    ActionChains(driver).move_to_element(button).click().perform()
                except Exception:
                    driver.execute_script("arguments[0].click();", button)

                print(f" -> Đã bấm 'Xem thêm' lần {click_idx + 1}")
                time.sleep(1.2)
                clicked = True
                break
            except (StaleElementReferenceException, WebDriverException):
                continue
            except Exception:
                continue

        if not clicked:
            print("[+] Không còn nút 'Xem thêm' hoặc đã tải hết.")
            break


def extract_reviews_from_page(driver) -> List[Review]:
    """Extracts all reviews currently rendered in the DOM."""
    from selenium.webdriver.common.by import By
    from selenium.common.exceptions import (
        StaleElementReferenceException,
        WebDriverException,
    )

    try:
        product_url = driver.current_url
    except Exception:
        product_url = None

    try:
        product_name = driver.find_element(By.CSS_SELECTOR, "h1").text.strip()
    except Exception:
        product_name = None

    print(f"\n[+] Sản phẩm: {product_name}")
    print(f"[+] URL: {product_url}")

    review_nodes = []
    try:
        review_nodes = driver.find_elements(
            By.CSS_SELECTOR, "div[id^='cmt-'][data-comment-id]"
        )
    except (StaleElementReferenceException, WebDriverException):
        pass

    print(f"[+] Tìm thấy: {len(review_nodes)} khối review")
    extracted_reviews = []

    for idx, node in enumerate(review_nodes):
        try:
            # 1. Username
            username = None
            try:
                u_node = node.find_element(
                    By.CSS_SELECTOR, ":scope > p.gap-10px.flex > span:nth-child(1)"
                )
                username = u_node.text.strip()
            except Exception:
                pass

            # 2. Review text
            text = None
            try:
                t_nodes = node.find_elements(
                    By.CSS_SELECTOR, ":scope > p:nth-child(3)"
                )
                for t_node in t_nodes:
                    val = t_node.text.strip()
                    if val:
                        text = val
                        break
            except Exception:
                pass

            if not text:
                continue

            # 3. Rating stars
            rating = None
            try:
                stars = node.find_elements(
                    By.CSS_SELECTOR, "i[class*='icon__row-6-col-1']"
                )
                if 1 <= len(stars) <= 5:
                    rating = len(stars)
            except Exception:
                pass

            review = Review(
                text=text,
                rating=rating,
                username=username,
                product_name=product_name,
                source_url=product_url,
            )
            extracted_reviews.append(review)
        except (StaleElementReferenceException, WebDriverException):
            continue
        except Exception:
            continue

    return extracted_reviews


def crawl_dmx_reviews(
    driver,
    max_clicks: int = 50,
    interactive: bool = True,
    url: Optional[str] = None,
) -> List[Review]:
    """Crawl reviews for the current or specified product page."""
    if url:
        driver.get(url)
        time.sleep(2)

    if interactive:
        print("\n" + "=" * 50)
        print("DIEN MAY XANH - TRÌNH THU THẬP ĐÁNH GIÁ")
        print("1. Điều hướng đến sản phẩm trên trình duyệt")
        print("2. Vượt Captcha nếu có")
        print("3. Cuộn đến khu vực đánh giá")
        print("=" * 50)
        input(">>> Nhấn Enter khi đã sẵn sàng để crawl...")
        time.sleep(1)

    load_more_reviews(driver, max_clicks=max_clicks)
    return extract_reviews_from_page(driver)


def save_to_csv(reviews: List[Review], filepath: str) -> None:
    """Appends crawled reviews to CSV with UTF-8-sig encoding."""
    import csv

    if not reviews:
        print("[!] Không có review nào để lưu.")
        return

    folder = os.path.dirname(filepath)
    if folder:
        os.makedirs(folder, exist_ok=True)

    file_exists = os.path.exists(filepath)
    fieldnames = ["noi_dung", "so_sao", "nguoi_danh_gia", "san_pham", "link"]

    mode = "a" if file_exists else "w"
    with open(filepath, mode=mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for rev in reviews:
            writer.writerow(rev.to_dict())

    print(f"[+] Đã lưu {len(reviews)} reviews vào '{filepath}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Crawler đánh giá sản phẩm Điện Máy Xanh."
    )
    parser.add_argument(
        "--output",
        default="data/raw/reviews_raw.csv",
        help="Đường dẫn file CSV kết quả (mặc định: data/raw/reviews_raw.csv)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="URL sản phẩm cần crawl (tùy chọn)",
    )
    parser.add_argument(
        "--max-clicks",
        type=int,
        default=50,
        help="Số lần bấm 'Xem thêm' tối đa (mặc định: 50)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Chạy Chrome ở chế độ không giao diện (headless)",
    )
    args = parser.parse_args()

    driver = create_driver(headless=args.headless)
    all_reviews: List[Review] = []

    try:
        if args.url:
            reviews = crawl_dmx_reviews(
                driver,
                max_clicks=args.max_clicks,
                interactive=False,
                url=args.url,
            )
            all_reviews.extend(reviews)
        else:
            while True:
                print("\n" + "=" * 40)
                print("Lựa chọn:")
                print(" - Nhấn Enter để crawl trang hiện tại")
                print(" - Nhập 0 để kết thúc và lưu file")
                print("=" * 40)
                choice = input(">>> ").strip()
                if choice == "0":
                    break
                reviews = crawl_dmx_reviews(
                    driver,
                    max_clicks=args.max_clicks,
                    interactive=True,
                )
                all_reviews.extend(reviews)
                print(f"[+] Sản phẩm này: {len(reviews)} review.")
                print(f"[+] Tổng cộng: {len(all_reviews)} review.")

        save_to_csv(all_reviews, args.output)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
