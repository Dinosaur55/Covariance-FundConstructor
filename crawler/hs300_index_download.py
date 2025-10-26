from stock_data_crawler_baostock import download_index_close_baostock

if __name__ == "__main__":
    # 与当前主流程一致的时间窗口
    download_index_close_baostock(
        index_code="sh.000300",  # 沪深300指数
        start_date="2014-10-01",
        end_date="2024-10-01",
        output_csv="hs300_index_column.csv",
    )
