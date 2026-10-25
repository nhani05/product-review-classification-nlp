import time

from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import (
    StaleElementReferenceException,
    WebDriverException,
)

from models.review import Review


def _first_text(element, selectors):
    """Trả về text đầu tiên tìm được theo danh sách CSS selector."""
    for selector in selectors:
        try:
            for node in element.find_elements(By.CSS_SELECTOR, selector):
                value = node.text.strip()
                if value:
                    return value
        except (StaleElementReferenceException, WebDriverException):
            continue
        except Exception:
            continue
    return None


def _get_review_text(item):
    """Lấy nội dung đánh giá từ cấu trúc HTML hiện tại của DMX."""

    try:
        nodes = item.find_elements(
            By.CSS_SELECTOR,
            ":scope > p:nth-child(3)"
        )

        for node in nodes:
            value = node.text.strip()
            if value:
                return value

    except (StaleElementReferenceException, WebDriverException):
        pass

    return None


def _get_username(item):
    """Lấy tên người đánh giá từ HTML của Điện Máy Xanh."""
    try:
        node = item.find_element(
            By.CSS_SELECTOR,
            ":scope > p.gap-10px.flex > span:nth-child(1)"
        )
        username = node.text.strip()
        return username if username else None

    except (StaleElementReferenceException, WebDriverException):
        return None
def _get_rating(item):
    """Đếm số sao đánh giá từ các icon col-1 của DMX."""
    try:
        stars = item.find_elements(
            By.CSS_SELECTOR,
            "i[class*='icon__row-6-col-1']"
        )
        count = len(stars)

        if 1 <= count <= 5:
            return count

    except (StaleElementReferenceException, WebDriverException):
        pass

    return None
def _get_review_items(driver):
    """Tìm các khối review theo cấu trúc HTML hiện tại của DMX."""
    try:
        return driver.find_elements(
            By.CSS_SELECTOR,
            "div[id^='cmt-'][data-comment-id]"
        )
    except (StaleElementReferenceException, WebDriverException):
        return []


def _load_more_reviews(driver, max_click=50):
    """Bấm Xem thêm tối đa max_click lần, dừng nếu không còn nút phù hợp."""
    print("\nĐang load thêm review...")

    for click_index in range(max_click):
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
                        element_id = element.id
                        if element_id not in seen_ids:
                            seen_ids.add(element_id)
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

                print(f"Đã bấm Xem thêm lần {click_index + 1}: {label}")
                time.sleep(1.2)
                clicked = True
                break

            except (StaleElementReferenceException, WebDriverException):
                continue
            except Exception:
                continue

        if not clicked:
            print("Không tìm thấy nút Xem thêm phù hợp hoặc không còn nút.")
            break


def crawl_dmx_reviews(driver):
    """Crawl review từ trang sản phẩm Điện Máy Xanh đang mở trong Selenium."""
    all_reviews = []

    print("\n========== DIEN MAY XANH - REVIEW CRAWLER ==========")
    print(
        """
Hãy thực hiện:
1. Mở sản phẩm cần crawl trên Điện Máy Xanh.
2. Vượt captcha nếu có.
3. Kéo xuống phần đánh giá và bấm xem tất cả đánh giá nếu cần.
4. Khi danh sách review đã hiện ra thì quay lại terminal.
"""
    )
    input(">>> Sẵn sàng crawl thì nhấn Enter...")
    time.sleep(2)

    # Bắt lỗi nếu Chrome đã đóng hoặc Selenium mất session.
    try:
        product_url = driver.current_url
    except WebDriverException as error:
        print("Không thể lấy URL vì Selenium đã mất kết nối với Chrome:", error)
        print("Hãy khởi động lại Chrome qua main.py rồi chạy crawler lại.")
        return all_reviews

    try:
        product_name = driver.find_element(By.CSS_SELECTOR, "h1").text.strip()
        if not product_name:
            product_name = None
    except Exception as error:
        print("Không lấy được tên sản phẩm:", error)
        product_name = None

    print("\nSản phẩm:", product_name)
    print("URL:", product_url)

    _load_more_reviews(driver)

    print("\nĐang lấy danh sách review...")
    review_elements = _get_review_items(driver)
    print("Tìm thấy:", len(review_elements), "khối review")

    # Không loại review trùng nội dung: mỗi khối HTML là một bản ghi riêng.
    for index, item in enumerate(review_elements):
        try:
            username = _get_username(item)
            text = _get_review_text(item)

            if not text:
                print(f"Review {index + 1} không tìm thấy nội dung, bỏ qua.")
                continue

            rating = _get_rating(item)

            review = Review(
                text=text,
                rating=rating,
                username=username,
                product_name=product_name,
                source_url=product_url,
            )
            all_reviews.append(review)

            print(f"\n[{len(all_reviews)}]")
            print("Người đánh giá:", username)
            print("Số sao:", rating)
            print("Nội dung:", text[:150])

        except (StaleElementReferenceException, WebDriverException) as error:
            print(f"Review {index + 1} bị stale hoặc mất kết nối:", error)
        except Exception as error:
            print(f"Lỗi review {index + 1}:", error)

    print("\n========== HOÀN THÀNH ==========")
    print("Tổng review crawl được:", len(all_reviews))
    return all_reviews
