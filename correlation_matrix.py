import pandas as pd
import numpy as np
from typing import Optional


def compute_profit_matrix_raw(input_csv: str, output_csv: Optional[str] = None) -> pd.DataFrame:
	"""
	从股票价格矩阵 CSV 计算收益矩阵 G。

		假设 CSV 首行为列名（pandas 自动识别表头），第 1 列为日期索引，其余列为各股票价格。

		预处理（构造形状为 T x N 的价格矩阵 Y）：
			- 使用第 1 列作为索引（日期）
			- 将其余列转为数值；无法转换的置为 NaN
			- 将非正价格（<= 0）置为 NaN（无法取对数）
	预处理(构造形状为 T x N 的价格矩阵 Y):
	  1) 删除第 1 行（股票代码）
	  2) 删除第 1 列（日期）
	  3) 将剩余单元格全部转为数值；无法转换的置为 NaN
	  4) 将非正价格(<= 0)置为 NaN(无法取对数)

	收益矩阵定义：
	  G[t, i] = log(Y[t+1, i]) - log(Y[t, i]),t=0..T-2(因此 G 形状为 (T-1) x N)
	  若 Y[t+1, i] 或 Y[t, i] 为 NaN, 则 G[t, i] 置为 NaN。

	参数：
		input_csv: 输入 CSV 文件路径
		output_csv: 若提供，则保存 G 的输出 CSV 路径

	返回：
		G 的 pandas DataFrame; 列名与价格矩阵一致（若可用）。
		行索引与原始日期索引右移后对齐（若有日期列），否则为整数索引。
	"""

	# 读取 CSV：自动识别表头，使用第 1 列作为索引（日期）
	df = pd.read_csv(input_csv, index_col=0)
	if df.shape[0] < 2 or df.shape[1] < 1:
		raise ValueError("输入 CSV 尺寸不符合预期；至少需要 T>=2（两期以上），N>=1（至少一只股票）")

	# 转为数值类型；不可转换的置为 NaN
	Y = df.apply(pd.to_numeric, errors='coerce')

	# 转为数值类型；不可转换的置为 NaN
	Y = Y.apply(pd.to_numeric, errors='coerce')

	# 非正价格无法取对数，置为 NaN
	Y = Y.mask(~(Y > 0))

	# 计算对数收益；Pandas 会自动传播 NaN
	logY = np.log(Y)
	G = logY.shift(-1, axis=0) - logY
	# 删除由于对齐产生的最后一行（对应最后一个日期），满足“删除最后一个日期”的要求
	G = G.iloc[:-1, :]

	# 若存在日期，则将 G 的索引对齐为“下一期”的日期（对应 Y 的 t->t+1）
	if len(df.index) >= 2:
		G.index = df.index[1:]

	if output_csv:
		G.to_csv(output_csv, index=True)

	return G


def compute_profit_matrix_full(input_csv: str, output_csv: Optional[str] = None) -> pd.DataFrame:
	"""
	从股票价格矩阵 CSV 计算“完整数据”版本的收益矩阵 G：
	- 自动识别表头，使用第 1 列作为日期索引
	- 将其余列转为数值，非正价格（<=0）与不可转换值置为 NaN
	- 在计算收益前，删除任何包含 NaN 的整列（仅保留 2014-2024 期间数据完整的股票）
	- 计算对数收益，同样删除最后一个日期对应的行

	注意：该函数与 compute_profit_matrix_raw 的区别在于“先列删缺失”，从而得到一个没有 NaN 的 Y 与 G。
	"""

	# 读取 CSV：自动识别表头，使用第 1 列作为索引（日期）
	df = pd.read_csv(input_csv, index_col=0)
	if df.shape[0] < 2 or df.shape[1] < 1:
		raise ValueError("输入 CSV 尺寸不符合预期；至少需要 T>=2（两期以上），N>=1（至少一只股票）")

	# 转为数值类型；不可转换的置为 NaN
	Y = df.apply(pd.to_numeric, errors='coerce')
	# 非正价格无法取对数，置为 NaN
	Y = Y.mask(~(Y > 0))

	# 删除含有任意 NaN 的整列，仅保留完全无缺失的股票列
	Y = Y.dropna(axis=1, how='any')

	if Y.shape[1] == 0:
		raise ValueError("删除缺失列后不再有可用股票列，请检查输入数据是否完整。")

	# 计算对数收益；并删除最后一行（最后一个日期）
	logY = np.log(Y)
	G = logY.shift(-1, axis=0) - logY
	G = G.iloc[:-1, :]

	# 将索引对齐到“下一期”的日期
	if len(df.index) >= 2:
		G.index = df.index[1:]

	if output_csv:
		G.to_csv(output_csv, index=True)

	return G


if __name__ == "__main__":
	# 示例：直接处理仓库中的文件
	input_path = "./stock_matrix/hs300_2014-2024_matrix.csv"
	# 原始版本（保留缺失）：
	output_path_raw = "profit_matrix_raw.csv"
	# 完整列版本（删除任何含缺失的股票列）：
	output_path_full = "profit_matrix_full.csv"

	try:
		G_raw = compute_profit_matrix_raw(input_path, output_csv=output_path_raw)
		print(f"已保存收益矩阵到 {output_path_raw}，形状: {G_raw.shape}")
		nan_ratio_raw = G_raw.isna().mean().mean()
		print(f"原始版本整体 NaN 占比: {nan_ratio_raw:.2%}")
	except Exception as e:
		print(f"计算原始收益矩阵失败: {e}")

	try:
		G_full = compute_profit_matrix_full(input_path, output_csv=output_path_full)
		print(f"已保存完整列收益矩阵到 {output_path_full}，形状: {G_full.shape}")
		nan_ratio_full = G_full.isna().mean().mean()
		print(f"完整列版本整体 NaN 占比: {nan_ratio_full:.2%}")
	except Exception as e:
		print(f"计算完整列收益矩阵失败: {e}")

