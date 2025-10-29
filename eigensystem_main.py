from corrkit import CorrKit

def main():
    # 示例：直接处理仓库中的文件
    input_path = "./stock_matrix/hs300_2014-2024_matrix.csv"
    # 收益矩阵输出：
    output_profit_full = "profit_matrix_full.csv"
    # 关联矩阵输出：
    output_corr_full = "correlation_matrix_full.csv"

    kit = CorrKit(input_path)

    # 计算收益矩阵
    R_full = kit.compute_profit_matrix(output_csv=output_profit_full, start_date="2022-10-01", end_date="2024-10-01")
    print(f"收益矩阵形状: {R_full.shape}")

    # 计算关联矩阵
    C_full = kit.compute_correlation_matrix(R_full, output_csv=output_corr_full)
    print(f"关联矩阵形状: {C_full.shape}")

    # 本征值分解与绘图
    vals, vecs = kit.spectral_decomposition(C_full)
    kit.plot_eigensystem(
        vals,
        vecs,
        T=R_full.shape[0],
        N=R_full.shape[1],
        eigenvalues_pdf="eigenvalues.png",
        eigenvectors_pdf="eigenvectors.png",
        eigenvectors_values_pdf="eigenvectors_values.png",
        eigenvec2_csv="eigenvector_2.csv",
        eigenvec3_csv="eigenvector_3.csv",
        asset_labels=C_full.columns,
    )
    print("已输出本征值与本征向量分布的对比图：eigenvalues.png, eigenvectors.png, eigenvectors_values.png；并保存 eigenvector_2.csv 与 eigenvector_3.csv")

if __name__ == "__main__":
    main()
