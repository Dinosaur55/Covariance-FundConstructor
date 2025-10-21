import baostock as bs
import pandas as pd
import time
import random
from tqdm import tqdm

def map_to_baostock_code(code):
    """把 600000 -> sh.600000, 000001 -> sz.000001"""
    code = code.strip()
    if code.startswith("6"):
        return "sh." + code
    else:
        return "sz." + code

def download_close_baostock(codes, start_date, end_date, output_csv="demo_close.csv", max_retries=3):
    lg = bs.login()
    print("登录 Baostock:", lg.error_code, lg.error_msg)

    per_stock = {}
    failed = []

    for raw_code in tqdm(codes, desc="下载演示"):
        bs_code = map_to_baostock_code(raw_code)
        success = False
        for attempt in range(max_retries):
            try:
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,close",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag="3"  # 不复权；如果要前复权用 "1"
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
            print(f"❌ {raw_code} 下载失败")
            failed.append(raw_code)
        time.sleep(random.uniform(0.3, 0.8))

    bs.logout()
    print("登出 Baostock")

    # 合并
    all_dates = None
    for df in per_stock.values():
        if all_dates is None:
            all_dates = df.index
        else:
            all_dates = all_dates.union(df.index)
    all_dates = sorted(all_dates)
    price_matrix = pd.DataFrame(index=all_dates)
    for code, df in per_stock.items():
        price_matrix = price_matrix.join(df, how="outer")
    price_matrix.sort_index(inplace=True)

    price_matrix.to_csv(output_csv, encoding="utf-8-sig")
    print("演示 CSV 已保存为", output_csv)
    print("形状：", price_matrix.shape)
    if failed:
        print("下载失败的股票：", failed)

    return price_matrix


if __name__ == "__main__":
    # 这里写前十大权重股示例代码：请你替换为真正的前十大权重股代码
    demo_codes = ["600000", "600519", "600036", "600276", "600104",
                  "600837", "600703", "601988", "600887", "601006"]
    df_demo = download_close_baostock(
        codes=demo_codes,
        start_date="2014-10-01",
        end_date="2024-10-01",
        output_csv="top10_demo.csv"
    )
    print(df_demo.head())