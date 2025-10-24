from corrkit import CorrKit

def main():
    # 示例：直接处理仓库中的文件
    input_path = "./stock_matrix/hs300_2014-2024_matrix.csv"
    # 收益矩阵输出：
    output_profit_full = "profit_matrix_full.csv"
    # 关联矩阵输出：
    output_corr_full = "correlation_matrix_full.csv"
    # 完整协方差矩阵输出：
    output_cov_full = "covariance_matrix_full.csv"
    # 去噪声后的协方差矩阵输出：
    output_cov_pr = "covariance_matrix_pr.csv"

    kit = CorrKit(input_path)

    # 计算收益矩阵
    R_full = kit.compute_profit_matrix(output_csv=output_profit_full, start_date="2022-10-01", end_date="2024-10-01")
    print(f"收益矩阵形状: {R_full.shape}")

    # 计算关联矩阵
    C_full = kit.compute_correlation_matrix(R_full, output_csv=output_corr_full)
    print(f"关联矩阵形状: {C_full.shape}")

    # 计算完整协方差矩阵
    CV_full = kit.compute_covariance_matrix_direct(R_full, output_csv=output_cov_full)
    print(f"完整协方差矩阵形状: {CV_full.shape}")

    # 计算去噪声后的协方差矩阵
    C_pr = kit.denoising(C_full, T=R_full.shape[0], N=R_full.shape[1], output_csv=None)
    D_full = kit.compute_std_matrix(R_full, output_csv=None)
    CV_pr = kit.compute_covariance_matrix(C_pr, D_full, output_csv=output_cov_pr)
    print(f"去噪声后协方差矩阵形状: {CV_pr.shape}")


if __name__ == "__main__":
    main()
