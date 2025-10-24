import pandas as pd
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

# Set LaTeX font for all text
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman",
    "font.size": 16,
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "text.latex.preamble": r"\usepackage{amsfonts}"
})

def compute_profit_matrix_raw(input_csv: str, output_csv: Optional[str] = None) -> pd.DataFrame:
	"""
	从股票价格矩阵 CSV 计算收益矩阵 R。

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
	  R[t, i] = log(Y[t+1, i]) - log(Y[t, i]),t=0..T-2(因此 R 形状为 (T-1) x N)
	  若 Y[t+1, i] 或 Y[t, i] 为 NaN, 则 R[t, i] 置为 NaN。

	参数：
		input_csv: 输入 CSV 文件路径
		output_csv: 若提供，则保存 G 的输出 CSV 路径

	返回：
		R 的 pandas DataFrame; 列名与价格矩阵一致（若可用）。
		行索引与原始日期索引右移后对齐（若有日期列），否则为整数索引。
	"""

	# 读取 CSV：自动识别表头，使用第 1 列作为索引（日期）
	df = pd.read_csv(input_csv, index_col=0)
	if df.shape[0] < 2 or df.shape[1] < 1:
		raise ValueError("输入 CSV 尺寸不符合预期；至少需要 T>=2（两期以上），N>=1（至少一只股票）")

	# 转为数值类型；不可转换的置为 NaN
	Y = df.apply(pd.to_numeric, errors='coerce')

	# 非正价格无法取对数，置为 NaN
	Y = Y.mask(~(Y > 0))

	# 计算对数收益；Pandas 会自动传播 NaN
	logY = np.log(Y)
	R = logY.shift(-1, axis=0) - logY
	# 删除由于对齐产生的最后一行（对应最后一个日期），满足“删除最后一个日期”的要求
	R = R.iloc[:-1, :]

	# 若存在日期，则将 R 的索引对齐为“下一期”的日期（对应 Y 的 t->t+1）
	if len(df.index) >= 2:
		R.index = df.index[1:]

	if output_csv:
		R.to_csv(output_csv, index=True)

	return R


def compute_profit_matrix(input_csv: str, output_csv: Optional[str] = None,
							start_date: Optional[str] = None,
							end_date: Optional[str] = None) -> pd.DataFrame:
	"""
	从股票价格矩阵 CSV 计算“完整数据”版本的收益矩阵 R：
	- 自动识别表头，使用第 1 列作为日期索引
	- 将其余列转为数值，非正价格（<=0）与不可转换值置为 NaN
	- 可选：先按时间窗口截取 [start_date, end_date]（包含端点）；
	- 在计算收益前，删除任何包含 NaN 的整列（仅保留该窗口内数据完整的股票）
	- 计算对数收益，同样删除最后一个日期对应的行

	参数：
		input_csv: 输入价格矩阵 CSV 路径
		output_csv: 若提供，则保存 R 的输出 CSV 路径
		start_date: 可选，起始日期（如 "2023-10-01"）；若为 None 则不设下界
		end_date: 可选，结束日期（如 "2024-10-01"）；若为 None 则不设上界

	注意：该函数与 compute_profit_matrix_raw 的区别在于“先列删缺失”，从而得到一个没有 NaN 的 Y 与 R；
	且若指定了时间窗口，会在“删列之前”先截取时间窗口。
	"""

	# 读取 CSV：自动识别表头，使用第 1 列作为索引（日期）
	df = pd.read_csv(input_csv, index_col=0)
	# 将索引解析为日期类型，便于时间切片（解析失败的保留为 NaT，但若需要切片建议使用标准日期格式）
	try:
		df.index = pd.to_datetime(df.index, errors='coerce')
	except Exception:
		pass

	# 可选：按时间窗口截取（在删列之前进行）
	if start_date is not None or end_date is not None:
		start_ts = pd.to_datetime(start_date) if start_date is not None else None
		end_ts = pd.to_datetime(end_date) if end_date is not None else None
		if start_ts is not None and end_ts is not None and start_ts > end_ts:
			raise ValueError("start_date 不应晚于 end_date")
		if start_ts is not None and end_ts is not None:
			df = df.loc[start_ts:end_ts]
		elif start_ts is not None:
			df = df.loc[start_ts:]
		elif end_ts is not None:
			df = df.loc[:end_ts]

	# 时间窗口截取后再做尺寸检查
	if df.shape[0] < 2 or df.shape[1] < 1:
		raise ValueError("选择的时间窗口或输入数据尺寸不符合预期；至少需要 T>=2，N>=1。")

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
	R = logY.shift(-1, axis=0) - logY
	R = R.iloc[:-1, :]

	# 将索引对齐到“下一期”的日期
	if len(df.index) >= 2:
		R.index = df.index[1:]

	if output_csv:
		R.to_csv(output_csv, index=True)

	return R


def compute_correlation_matrix(R: pd.DataFrame, output_csv: Optional[str] = None) -> pd.DataFrame:
	"""
	基于 R_full 按列归一化后计算关联矩阵 C，G[t,i] = (r_{t,i} - <r_i>)/sigma_i，C = (1/T) G^T G。
	"""

	T = R.shape[0]
	mu = R.mean(axis=0)
	sigma = R.std(axis=0, ddof=0) # 使用总体标准差(ddof=0)，为了让G矩阵中每个元素方差为1
	nonzero = sigma != 0
	R = R.loc[:, nonzero]
	mu = mu[nonzero]
	sigma = sigma[nonzero]
	G = (R - mu) / sigma
	C = (G.T @ G) / T
	if output_csv:
		C.to_csv(output_csv, index=True)
	return C


def compute_covariance_matrix_direct(R: pd.DataFrame, output_csv: Optional[str] = None) -> pd.DataFrame:
	"""
	基于 R_full 按列将期望值置零后计算协方差矩阵 CV，g[t,i] = (r_{t,i} - <r_i>)，CV = (1/T) g^T g。
	"""

	T = R.shape[0]
	mu = R.mean(axis=0)
	g = (R - mu)
	C = (g.T @ g) / T
	if output_csv:
		C.to_csv(output_csv, index=True)
	return C


def spectral_decomposition(C: pd.DataFrame):
	"""
	对关联矩阵 C 进行本征值分解，返回按“从大到小”排序的本征值与对应本征向量。
	"""
	# eigh 返回对称矩阵的本征系统，默认本征值从小到大
	vals, vecs = np.linalg.eigh(C.values)
	# 改为从大到小排序，并同步重排本征向量的列
	idx_desc = np.argsort(vals)[::-1]
	vals_sorted = vals[idx_desc]
	vecs_sorted = vecs[:, idx_desc]
	return vals_sorted, vecs_sorted


def denoising(C: pd.DataFrame, T: int, N: int, output_csv: Optional[str] = None) -> pd.DataFrame:
	"""
	基于随机矩阵理论(RMT)的去噪：只保留大于 λ+ 的本征值及其对应本征向量，重构去噪后的关联矩阵 C_pr。

	参数：
		C: 输入的关联矩阵（对称，形状 N x N）
		T: 时间长度（样本期数）
		N: 资产数量（矩阵维度）

	函数内打印：保留的全部本征值列表与阈值 λ+ 的数值。

	返回：
		C_pr: 去噪后的关联矩阵（pandas DataFrame，索引与列与 C 对齐）。
	"""
	# 计算阈值 λ+
	Q = T / N
	lam_plus = 1 + 1 / Q + 2 * np.sqrt(1 / Q)

	# 使用已有的谱分解函数（本征值已按从大到小排序）
	vals_desc, vecs_desc = spectral_decomposition(C)
	mask = vals_desc > lam_plus
	vals_kept = vals_desc[mask]
	if vals_kept.size == 0:
		print(f"去噪：无本征值大于 λ+={lam_plus:.6f}，返回零矩阵。")
		zeros = np.zeros_like(C.values)
		return pd.DataFrame(zeros, index=C.index, columns=C.columns)

	V = vecs_desc[:, mask]
	C_pr_np = V @ np.diag(vals_kept) @ V.T
	# 数值对称化
	C_pr_np = 0.5 * (C_pr_np + C_pr_np.T)
	# 转为带标签的 DataFrame，便于保存与后续按标签运算
	C_pr = pd.DataFrame(C_pr_np, index=C.index, columns=C.columns)

	# 可选保存 CSV
	if output_csv:
		C_pr.to_csv(output_csv, index=True)

	# 打印信息
	print(f"λ+ = {lam_plus:.6f}，保留本征值: {np.array2string(vals_kept, precision=6, separator=', ')}")

	return C_pr


def compute_std_matrix(R: pd.DataFrame, output_csv: Optional[str] = None) -> pd.DataFrame:
	"""
	构造对角矩阵 D，其对角线为收益矩阵 R 各列的标准差（总体标准差，ddof=0）。
	"""
	# 计算每列标准差（总体标准差，使得标准化后方差为 1）
	sigma = R.std(axis=0, ddof=0)
	# 构造对角矩阵，并保留标签（索引/列名）
	D = pd.DataFrame(np.diag(sigma.values), index=sigma.index, columns=sigma.index)
	if output_csv:
		D.to_csv(output_csv, index=True)
	return D


def compute_covariance_matrix(C: pd.DataFrame, D: pd.DataFrame, output_csv: Optional[str] = None) -> pd.DataFrame:
	"""
	基于关联矩阵 C (可能是去噪声后的) 和标准差对角矩阵 D 计算协方差矩阵 CV，CV = D C D，其中 D 为对角标准差矩阵。
	"""
	CV = D @ C @ D
	if output_csv:
		CV.to_csv(output_csv, index=True)
	return CV


def plot_eigensystem(eigvals: np.ndarray, eigvecs: np.ndarray, T: int, N: int,
							eigenvalues_pdf: str = "eigenvalues.pdf",
							eigenvectors_pdf: str = "eigenvectors.pdf") -> None:
	"""
	绘制谱系结果（图中文字为英文，代码注释为中文）：
	  - 本征值直方图（与随机矩阵理论 MP 密度对比，并标注 λ±）
	  - 上排：最大的 3 个本征向量元素分布；下排：第 50、100、200 个本征向量元素分布
	说明：不生成随机矩阵，仅叠加理论曲线。
	"""
	# 计算 Q 与理论端点 λ±
	Q = T / N
	lam_minus = 1 + 1 / Q - 2 * np.sqrt(1 / Q)
	lam_plus = 1 + 1 / Q + 2 * np.sqrt(1 / Q)

	# 本征值分布图：以 0.05 为分箱宽度，横轴从 0 到“略高于第 4 大本征值”，不展示最大本征值影响
	plt.figure(figsize=(7, 5))
	bin_width = 0.05
	start = 0.0
	if len(eigvals) >= 4:
		upper = float(eigvals[3]) * 1.02  # 微微露出第 4 大本征值
	else:
		upper = float(np.max(eigvals))
	# 防止上界过小
	upper = max(upper, start + bin_width)
	# 将上界对齐到 0.05 网格
	right = np.ceil(upper / bin_width) * bin_width
	bins = np.arange(start, right + bin_width, bin_width)
	plt.hist(
		eigvals,
		bins=bins,
		density=True,
		alpha=0.7,
		color="#4C78A8",
		edgecolor="white",
		label="Empirical"
	)

	# 理论 MP 密度（带根号的正确公式）
	lam_grid = np.linspace(max(lam_minus, 1e-9), lam_plus, 1000)
	support_mask = (lam_grid >= lam_minus) & (lam_grid <= lam_plus)
	mp_density = np.zeros_like(lam_grid)
	mp_density[support_mask] = (Q / (2 * np.pi)) * (
		np.sqrt((lam_plus - lam_grid[support_mask]) * (lam_grid[support_mask] - lam_minus)) / lam_grid[support_mask]
	)
	plt.plot(lam_grid, mp_density, "-", color="#F58518", linewidth=2.2, label="RMT")

	# 不再绘制竖线；仅在图例中展示 λ± 的数值（用空数据曲线生成图例条目）
	plt.plot([], [], color="#E45756", linestyle="--", linewidth=1.5,
			 label=rf"$\lambda_- = {lam_minus:.3f}$")
	plt.plot([], [], color="#72B7B2", linestyle="--", linewidth=1.5,
			 label=rf"$\lambda_+ = {lam_plus:.3f}$")
	plt.xlabel(r"$\lambda$")
	plt.ylabel(r"$p(\lambda)$")
	plt.title("Eigenvalue distribution: empirical vs RMT")
	# x 轴范围设为 [0, 略高于第 4 大本征值]
	plt.xlim(start, right)
	plt.legend()
	plt.tight_layout()
	plt.savefig(eigenvalues_pdf)
	plt.close()

	# 本征向量元素分布：上排=最大3个；下排=第50/100/200个（均以“从大到小排序”的序号定义）
	# 注意：此时 eigvals/ eigvecs 已按从大到小排列，因此索引 0,1,2 即为最大三个
	top3_idx = [0, 1, 2]
	requested_positions = [50, 100, 200]
	valid_positions = [p for p in requested_positions if 1 <= p <= len(eigvals)]
	# 将 1-based 的第 p 个，转换为 0-based 的列索引
	bottom_idx = [p - 1 for p in valid_positions]
	order_indices = top3_idx + bottom_idx

	# 标准正态对比曲线
	x_grid = np.linspace(-4, 4, 801)
	normal_pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-(x_grid ** 2) / 2)

	fig, axes = plt.subplots(2, 3, figsize=(11, 7))
	for k, ax in enumerate(axes.flat):
		if k < len(order_indices):
			col_idx = order_indices[k]
			vec = eigvecs[:, col_idx]
			# 若为第一大本征向量且整体符号为负，则翻转符号（数学上等价）
			if col_idx == 0 and (np.all(vec <= 0) or np.mean(vec) < 0):
				vec = -vec
			# 根据随机矩阵理论，sqrt(N) * u_j ~ N(0,1)，因此在绘图前进行缩放
			vec_scaled = vec * np.sqrt(N)
			ax.hist(
				vec_scaled,
				bins=50,
				density=True,
				alpha=0.9,
				color="#2E7D32",
				edgecolor="#1B5E20",
				label="Empirical"
			)
			ax.plot(x_grid, normal_pdf, "-", color="#EECA3B", linewidth=2.2, label="RMT")
			ev = eigvals[col_idx]

			if k < 3:
				i_label = k + 1
				title = rf"$\lambda_{{{i_label}}} = {ev:.3f}$"
			else:
				i_label = valid_positions[k - 3] if (k - 3) < len(valid_positions) else (k + 1)
				title = rf"$\lambda_{{{i_label}}} = {ev:.3f}$"
			ax.set_title(title)
			ax.set_xlim(-4, 4)
			ax.set_xlabel(rf'$\mathbf{{u}}_{{{i_label}}}$')
			if k % 3 == 0:
				ax.set_ylabel(rf'$p(\mathbf{{u}}_{{{i_label}}})$')
		else:
			ax.axis('off')
			
		# 仅左上角子图保留图例，其余子图不显示
		if k == 0:
			ax.legend(fontsize=12)

	plt.tight_layout()
	plt.savefig(eigenvectors_pdf)
	plt.close()


if __name__ == "__main__":
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
	# 去噪声后的关联矩阵输出：
	output_corr_pr = "correlation_matrix_pr.csv"
	# 标准差对角矩阵输出：
	output_std_full = "std_matrix_full.csv"

	try:
		# 计算收益矩阵
		R_full = compute_profit_matrix(input_path, output_csv=output_profit_full, start_date="2022-10-01", end_date="2024-10-01")
		print(f"收益矩阵形状: {R_full.shape}")
		# 计算关联矩阵
		C_full = compute_correlation_matrix(R_full, output_csv=output_corr_full)
		print(f"关联矩阵形状: {C_full.shape}")
		# 本征值分解与绘图
		vals, vecs = spectral_decomposition(C_full)
		plot_eigensystem(vals, vecs, T=R_full.shape[0], N=R_full.shape[1],
					  eigenvalues_pdf="eigenvalues.pdf",
					  eigenvectors_pdf="eigenvectors.pdf")
		print("已输出本征值与本征向量分布的对比图：eigenvalues.pdf, eigenvectors.pdf")

		# 计算完整协方差矩阵
		CV_full = compute_covariance_matrix_direct(R_full, output_csv=output_cov_full)
		print(f"完整协方差矩阵形状: {CV_full.shape}")
		# 计算去噪声后的协方差矩阵
		C_pr = denoising(C_full, T=R_full.shape[0], N=R_full.shape[1], output_csv=None)
		D_full = compute_std_matrix(R_full, output_csv=None)
		CV_pr = compute_covariance_matrix(C_pr, D_full, output_csv=output_cov_pr)
		print(f"去噪声后协方差矩阵形状: {CV_pr.shape}")

	except Exception as e:
		print(f"关联矩阵分析失败: {e}")

