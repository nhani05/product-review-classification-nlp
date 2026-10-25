import os
import pandas as pd


def save_to_csv(reviews, filename):
    if not reviews:
        print("\nKhông có review nào để lưu.")
        return

    data = []

    for review in reviews:
        data.append(review.to_dict())

    df_new = pd.DataFrame(data)

    # =====================================================
    # LÀM SẠCH NỘI DUNG
    # =====================================================
    if "noi_dung" in df_new.columns:
        df_new["noi_dung"] = (
            df_new["noi_dung"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        # Chỉ bỏ review rỗng.
        # KHÔNG xóa review có nội dung trùng nhau.
        df_new = df_new[df_new["noi_dung"] != ""]

    folder = os.path.dirname(filename)

    if folder:
        os.makedirs(folder, exist_ok=True)

    # =====================================================
    # NẾU FILE ĐÃ TỒN TẠI -> ĐỌC DỮ LIỆU CŨ VÀ NỐI THÊM
    # =====================================================
    if os.path.exists(filename):
        try:
            df_old = pd.read_csv(
                filename,
                encoding="utf-8-sig"
            )

            df = pd.concat(
                [df_old, df_new],
                ignore_index=True
            )

        except Exception as e:
            print("Không đọc được file cũ:", e)
            df = df_new
    else:
        df = df_new

    # =====================================================
    # KHÔNG LOẠI REVIEW TRÙNG
    # =====================================================
    # Cố ý giữ nguyên các review trùng nội dung, trùng tên,
    # trùng số sao... vì mỗi review crawl được đều được tính
    # là một bản ghi dữ liệu.

    # =====================================================
    # LƯU UTF-8
    # =====================================================
    df.to_csv(
        filename,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== DONE ==========")
    print(df.tail())

    print(f"\nReview vừa crawl: {len(df_new)}")
    print(f"Tổng review trong file: {len(df)}")
    print(f"File: {filename}")
