import baostock as bs
import pandas as pd
import time
import random
from tqdm import tqdm

def download_index_close_baostock(index_code: str = "sh.000300",
                                  start_date: str = "2022-10-01",
                                  end_date: str = "2024-10-01",
                                  output_csv: str = "hs300_index_close.csv",
                                  max_retries: int = 3) -> pd.DataFrame:
    """
    下载单一指数(日线收盘价)，输出与股票价格矩阵一致的 CSV 格式：
    - 第 1 列为日期索引
    - 其余列为价格列（这里只有 1 列，列名为 index_code）

    参数：
      index_code: 指数代码（Baostock 格式，沪深300 为 "sh.000300"）
      start_date: 起始日期，YYYY-MM-DD
      end_date:   结束日期，YYYY-MM-DD
      output_csv: 输出 CSV 路径
      max_retries: 失败重试次数
    返回：
      DataFrame，index=日期，列=[index_code]
    """
    lg = bs.login()
    print("登录 Baostock:", lg.error_code, lg.error_msg)

    success = False
    last_err = None
    for attempt in range(max_retries):
        try:
            rs = bs.query_history_k_data_plus(
                index_code,
                "date,close",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3"  # 不复权
            )
            if rs.error_code != "0":
                raise RuntimeError(f"baostock error {rs.error_code}: {rs.error_msg}")

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            df = pd.DataFrame(data_list, columns=rs.fields)
            if df.empty:
                raise RuntimeError("empty result")

            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df = df.rename(columns={"close": index_code})
            df = df[[index_code]]
            success = True
            break
        except Exception as e:
            last_err = e
            wait = random.uniform(1.0, 2.0)
            print(f"⚠️ 指数 {index_code} 第 {attempt+1} 次失败 ({e})，{wait:.1f}s 后重试")
            time.sleep(wait)

    bs.logout()
    print("登出 Baostock")

    if not success:
        raise RuntimeError(f"下载指数 {index_code} 失败: {last_err}")

    df.sort_index(inplace=True)
    df.to_csv(output_csv, index=True)
    print(f"✅ 指数数据已保存为 {output_csv}，形状: {df.shape}")
    return df

def download_close_baostock(codes, start_date, end_date, output_csv="close_price_matrix.csv", max_retries=3):
    lg = bs.login()
    print("登录 Baostock:", lg.error_code, lg.error_msg)

    per_stock = {}
    failed = []

    for raw_code in tqdm(codes, desc="下载中"):
        success = False
        for attempt in range(max_retries):
            try:
                rs = bs.query_history_k_data_plus(
                    raw_code,
                    "date,close",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag="3"
                )
                if rs.error_code != "0":
                    raise RuntimeError(f"baostock error {rs.error_code}: {rs.error_msg}")

                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())
                df = pd.DataFrame(data_list, columns=rs.fields)
                if df.empty:
                    raise RuntimeError("empty result")

                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                df = df.rename(columns={"close": raw_code})
                per_stock[raw_code] = df[[raw_code]]
                success = True
                break
            except Exception as e:
                wait = random.uniform(1.0, 3.0)
                print(f"⚠️ {raw_code} 第 {attempt+1} 次失败 ({e})，{wait:.1f}s 后重试")
                time.sleep(wait)
        if not success:
            failed.append(raw_code)
        time.sleep(random.uniform(0.3, 0.8))

    bs.logout()
    print("登出 Baostock")

    # 合并
    all_dates = sorted(set().union(*[df.index for df in per_stock.values()]))
    price_matrix = pd.DataFrame(index=all_dates)
    for code, df in per_stock.items():
        price_matrix = price_matrix.join(df, how="outer")
    price_matrix.sort_index(inplace=True)

    price_matrix.to_csv(output_csv, encoding="utf-8-sig")
    print(f"✅ 数据已保存为 {output_csv}，形状: {price_matrix.shape}")
    if failed:
        print("下载失败的股票：", failed)
    return price_matrix


if __name__ == "__main__":
    # 获取沪深300成分股列表
    lg = bs.login()
    rs = bs.query_hs300_stocks()
    hs300 = []
    while rs.next():
        hs300.append(rs.get_row_data()[1])  # 第二列是股票代码
    bs.logout()

    print(f"沪深300成分股数量: {len(hs300)}")

    print(hs300)

    df_all = download_close_baostock(
        codes=hs300,
        start_date="2014-10-01",
        end_date="2024-10-01",
        output_csv="hs300_close_matrix.csv"
    )
