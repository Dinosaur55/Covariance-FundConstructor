from corrkit import CorrKit

def main():
    # 示例：直接处理仓库中的文件
    input_path = "./stock_matrix/hs300_2014-2024_matrix.csv"
    index_path = "./stock_matrix/hs300_index_2014-2024_column.csv"
    # 收益矩阵输出：
    output_profit_full = "profit_matrix_full.csv"
    # 精选收益矩阵输出：
    output_profit_clean = "profit_matrix_clean.csv"
    # 关联矩阵输出：
    output_corr_clean = "correlation_matrix_clean.csv"
    # 完整协方差矩阵输出：
    output_cov_clean = "covariance_matrix_clean.csv"
    # 去噪声后的协方差矩阵输出：
    output_cov_pr = "covariance_matrix_pr.csv"

    kit = CorrKit(input_path)

    # 计算收益矩阵
    R_full = kit.compute_profit_matrix(output_csv=output_profit_full, start_date="2020-10-01", end_date="2023-10-01")
    print(f"收益矩阵形状: {R_full.shape}")

    # 基于 (σ, μ) 的 Pareto 初筛，得到精选收益矩阵
    R_clean = kit.clean_profit_matrix(R_full, output_csv=output_profit_clean,
                                      plot_path="pareto_mu_sigma.pdf", target_k=40)
    print(f"精选收益矩阵形状: {R_clean.shape}")

    # 使用精选后的收益矩阵计算关联矩阵
    C_clean = kit.compute_correlation_matrix(R_clean, output_csv=output_corr_clean)
    print(f"关联矩阵形状: {C_clean.shape}")

    # 计算完整协方差矩阵
    CV_clean = kit.compute_covariance_matrix_direct(R_clean, output_csv=output_cov_clean)
    print(f"完整协方差矩阵形状: {CV_clean.shape}")

    # 计算去噪声后的协方差矩阵
    C_pr = kit.denoising(C_clean, T=R_clean.shape[0], N=R_clean.shape[1], output_csv=None)
    D_clean = kit.compute_std_matrix(R_clean, output_csv=None)
    CV_pr = kit.compute_covariance_matrix(C_pr, D_clean, output_csv=output_cov_pr)
    print(f"去噪声后协方差矩阵形状: {CV_pr.shape}")

    # 计算相对沪深300指数的 Beta 因子
    kit_index = CorrKit(index_path)
    R_hs300 = kit_index.compute_profit_matrix(output_csv=None, start_date="2020-10-01", end_date="2023-10-01")
    beta_vec = kit.compute_beta_factors(R_hs300, R_clean)
    # 基于 beta 与协方差矩阵，计算最优权重（格式与 beta_vec 一致）
    omega_vec = kit.optimize_portfolio(beta_vec, CV_clean)
    print("最优权重向量（omega）：")
    print(omega_vec)


if __name__ == "__main__":
    main()
