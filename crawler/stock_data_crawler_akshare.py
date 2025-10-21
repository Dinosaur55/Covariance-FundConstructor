# import akshare as ak
# import pandas as pd
# import time
# import random
# from tqdm import tqdm


# def safe_download(code, start_date, end_date, max_retries=3):
#     """自动重试的下载函数"""
#     for attempt in range(max_retries):
#         try:
#             df = ak.stock_zh_a_hist(
#                 symbol=code,
#                 period="daily",
#                 start_date=start_date,
#                 end_date=end_date,
#                 adjust="qfq"
#             )
#             if df is not None and not df.empty:
#                 return df
#         except Exception as e:
#             wait = random.uniform(1, 3)
#             print(f"⚠️ {code} 第 {attempt+1} 次下载失败 ({e})，{wait:.1f}s 后重试...")
#             time.sleep(wait)
#     print(f"❌ {code} 多次重试仍失败，跳过。")
#     return None


# def download_close_prices(index_code="000001", start_date="20141001", end_date="20241001", output_csv="close_price_matrix.csv"):
#     print("正在获取上证指数成分股列表...")
#     cons = ak.index_stock_cons(symbol=index_code)
#     stock_list = cons["品种代码"].tolist()
#     print(f"共获取 {len(stock_list)} 只成分股。")

#     price_matrix = pd.DataFrame()
#     failed_list = []

#     for code in tqdm(stock_list, desc="下载收盘价中"):
#         df = safe_download(code, start_date, end_date)
#         if df is None:
#             failed_list.append(code)
#             continue

#         df = df[["日期", "收盘"]].rename(columns={"日期": "date", "收盘": code})
#         df["date"] = pd.to_datetime(df["date"])
#         df.set_index("date", inplace=True)

#         price_matrix = price_matrix.join(df, how="outer") if not price_matrix.empty else df

#         # ✅ 随机延迟以防封 IP
#         time.sleep(random.uniform(0.5, 1.5))

#     price_matrix.sort_index(inplace=True)
#     price_matrix = price_matrix.fillna(method="ffill")  # 停牌时沿用前值
#     price_matrix.to_csv(output_csv, encoding="utf-8-sig")

#     print(f"\n✅ 已保存为 {output_csv}")
#     print(f"数据维度: {price_matrix.shape}")

#     if failed_list:
#         print(f"⚠️ 以下股票下载失败共 {len(failed_list)} 只：")
#         print(failed_list)

#     return price_matrix


# # 示例调用
# if __name__ == "__main__":
#     h_matrix = download_close_prices(
#         index_code="000001",  # 上证指数
#         start_date="20141001",
#         end_date="20241001",
#         output_csv="shanghai_close_prices.csv"
#     )


import akshare as ak
import pandas as pd
from tqdm import tqdm


def download_close_prices(index_code="000001", start_date="20141001", end_date="20241001", output_csv="close_price_matrix.csv"):
    """
    从上证指数成分股下载每日收盘价矩阵 h_it
    参数:
        index_code : str, 指数代码（上证指数: "000001"）
        start_date : str, 起始日期, 格式 "YYYYMMDD"
        end_date   : str, 结束日期
        output_csv : str, 输出 CSV 文件名
    """

    print("正在获取上证指数成分股列表...")
    cons = ak.index_stock_cons(symbol=index_code)
    stock_list = cons["品种代码"].tolist()
    print(f"共获取 {len(stock_list)} 只成分股。")

    price_matrix = pd.DataFrame()

    for code in tqdm(stock_list, desc="下载收盘价中"):
        try:
            # AkShare 要求股票代码加上市场后缀
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            df = df[["日期", "收盘"]].rename(columns={"日期": "date", "收盘": code})
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

            if price_matrix.empty:
                price_matrix = df
            else:
                price_matrix = price_matrix.join(df, how="outer")

        except Exception as e:
            print(f"下载 {code} 失败: {e}")

    # 按日期排序并填充缺失
    price_matrix.sort_index(inplace=True)
    price_matrix = price_matrix.fillna(method="ffill")  # 停牌时沿用前值

    price_matrix.to_csv(output_csv, encoding="utf-8-sig")
    print(f"\n✅ 已保存为 {output_csv}")
    print(f"数据维度: {price_matrix.shape}")
    return price_matrix

# 示例调用
if __name__ == "__main__":
    h_matrix = download_close_prices(
        index_code="000001",  # 上证指数
        start_date="20141001",
        end_date="20241001",
        output_csv="shanghai_close_prices.csv"
    )
