from corrkit import CorrKit

def main():
    # 示例：直接处理仓库中的文件
    input_path = "./stock_matrix/hs300_2014-2024_matrix.csv"
    index_path = "./stock_matrix/hs300_index_2014-2024_column.csv"
    # # 精选收益矩阵输出：
    # output_profit_clean = "profit_matrix_clean.csv"
    # # 关联矩阵输出：
    # output_corr_clean = "correlation_matrix_clean.csv"
    # 去噪声后的关联矩阵输出：
    output_corr_pr = "correlation_matrix_pr.csv"
    # 完整协方差矩阵输出：
    output_cov_clean = "covariance_matrix_clean.csv"
    # 去噪声后的协方差矩阵输出：
    output_cov_pr = "covariance_matrix_pr.csv"

    kit = CorrKit(input_path)

    # 计算收益矩阵
    R_full = kit.compute_profit_matrix(output_csv=None, start_date="2020-10-01", end_date="2023-10-01")
    print(f"收益矩阵形状: {R_full.shape}")

    # 基于 (σ, μ) 的 Pareto 初筛，得到精选收益矩阵
    R_clean = kit.clean_profit_matrix(R_full, output_csv=None,
                                      plot_path="pareto_mu_sigma.png", target_k=40)
    print(f"精选收益矩阵形状: {R_clean.shape}")
    print("已输出收益率均值-标准差散点图为 pareto_mu_sigma.png")

    # 使用精选后的收益矩阵计算关联矩阵
    C_clean = kit.compute_correlation_matrix(R_clean, output_csv=None)

    # 计算完整协方差矩阵
    CV_clean = kit.compute_covariance_matrix_direct(R_clean, output_csv=output_cov_clean)
    print(f"完整协方差矩阵形状: {CV_clean.shape}")

    # 计算去噪声后的协方差矩阵
    C_pr = kit.denoising(C_clean, T=R_clean.shape[0], N=R_clean.shape[1], output_csv=output_corr_pr)
    D_clean = kit.compute_std_matrix(R_clean, output_csv=None)
    CV_pr = kit.compute_covariance_matrix(C_pr, D_clean, output_csv=output_cov_pr)
    print(f"去噪声后协方差矩阵形状: {CV_pr.shape}")

    # 计算相对沪深300指数的 Beta 因子（训练窗）
    kit_index = CorrKit(index_path)
    R_hs300 = kit_index.compute_profit_matrix(output_csv=None, start_date="2020-10-01", end_date="2023-10-01")
    beta_vec = kit.compute_beta_factors(R_hs300, R_clean)
    # 基于 beta 与协方差矩阵，计算最优权重（格式与 beta_vec 一致）
    omega_vec = kit.optimize_portfolio(beta_vec, CV_clean)
    omega_pr_vec = kit.optimize_portfolio(beta_vec, CV_pr)
    # print("omega_vec (CV):", list(omega_vec), " | omega_pr_vec (CV_pr):", list(omega_pr_vec))
    print("ETF已生成")

    # 回测：设初始净值为 100，从 2023-10-01 之后的第一个交易日至 2024-09-30
    bt = kit.backtest_buy_and_hold(
        omega=omega_vec,
        stock_csv=input_path,
        index_csv=index_path,
        start_date="2023-10-02",
        end_date="2024-09-30",
        plot_path="backtest_fund_vs_hs300.png",
        omega_pr=omega_pr_vec,
    )
    print("回测完成，曲线已保存为 backtest_fund_vs_hs300.png")

    # 输出 50/100/150/200 日夏普（普通收益率），以及均值与标准差 - Fund (CV)
    print("基金滚动夏普比率（普通收益率）- Fund (CV)：")
    for wlen in [50, 100, 150, 200]:
        mu, sigma, sh = bt['sharpe'][wlen]
        print(f"{wlen} 日: mu={mu:.6f}, sigma={sigma:.6f}, Sharpe={sh:.6f}")
    # 输出相对沪深300的超额收益夏普 - Fund (CV)
    print("相对沪深300超额收益的滚动夏普 - Fund (CV)：")
    for wlen in [50, 100, 150, 200]:
        mu_e, sigma_e, sh_e = bt['sharpe_excess'][wlen]
        print(f"{wlen} 日: mu_excess={mu_e:.6f}, sigma_excess={sigma_e:.6f}, Sharpe_excess={sh_e:.6f}")

    # Fund (CV_pr)
    print("基金滚动夏普比率（普通收益率）- Fund (CV_pr)：")
    for wlen in [50, 100, 150, 200]:
        mu2, sigma2, sh2 = bt['sharpe_pr'][wlen]
        print(f"{wlen} 日: mu={mu2:.6f}, sigma={sigma2:.6f}, Sharpe={sh2:.6f}")
    print("相对沪深300超额收益的滚动夏普 - Fund (CV_pr)：")
    for wlen in [50, 100, 150, 200]:
        mu2e, sigma2e, sh2e = bt['sharpe_excess_pr'][wlen]
        print(f"{wlen} 日: mu_excess={mu2e:.6f}, sigma_excess={sigma2e:.6f}, Sharpe_excess={sh2e:.6f}")

if __name__ == "__main__":
    main()
