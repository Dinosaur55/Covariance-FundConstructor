import pandas as pd
import numpy as np
from typing import Optional, Union
import matplotlib.pyplot as plt
import shutil

# Set LaTeX font for all text
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman",
    "font.size": 16,
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "text.latex.preamble": r"\usepackage{amsfonts}"
})

if shutil.which("latex") is None:
    plt.rcParams.update({"text.usetex": False})

class CorrKit:
    """
    一个紧凑的相关性/协方差与谱分析工具类。

    使用 input_path（股票价格矩阵 CSV 路径）进行初始化。类中的方法与原先的同名函数保持相同的功能与行为；
    对于原先需要 input_csv 的方法，如果省略该参数，则默认使用 self.input_path。
    """

    def __init__(self, input_path: str):
        self.input_path = input_path

    def compute_profit_matrix_raw(self, input_csv: Optional[str] = None, output_csv: Optional[str] = None) -> pd.DataFrame:
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
        input_csv = input_csv or self.input_path
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

    def compute_profit_matrix(self, input_csv: Optional[str] = None, output_csv: Optional[str] = None,
                              start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
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
        input_csv = input_csv or self.input_path
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

    def clean_profit_matrix(self, R: pd.DataFrame, output_csv: Optional[str] = None,
                            plot_path: str = "pareto_mu_sigma.pdf",
                            target_k: Optional[int] = 40) -> pd.DataFrame:
        """
        基于 Pareto 思想在 (σ, μ) 平面进行筛选，并按夏普比率补齐到目标数量，返回精选后的收益矩阵 R_clean。

        - 指标定义：按列计算收益均值 μ_i 与标准差 σ_i（总体标准差 ddof=0）。
        - 优劣方向：在“收益更高（μ 大）且波动更低（σ 小）”的方向更优。
        - 支配关系定义（j 支配 i）：μ_j ≥ μ_i 且 σ_j ≤ σ_i，且至少一项严格不等。
    - 选择规则（第一步，严格 Pareto）：仅保留“未被任何股票支配”的股票（即 dominated_count == 0）。
    - 选择规则（第二步，若目标数 target_k 指定且尚不足）：对未入选的其他股票，按夏普比率 μ/σ 从高到低补齐至 target_k。
        - 输出：
            1) 以紫色标注被 ≤1 支配的股票，其余为黑色的 (σ, μ) 散点图，并保存到 plot_path。
            2) 返回仅包含所选股票列的 R_clean（列顺序保持与 R 一致）。
            3) 若提供 output_csv，则将 R_clean 保存为 CSV。
        """
        # 计算每只股票的收益均值与标准差
        mu = R.mean(axis=0)
        sigma = R.std(axis=0, ddof=0)
        stats = pd.DataFrame({"mu": mu, "sigma": sigma})

        # 计算每只股票被“更高收益且更低波动”支配的次数（O(N^2)）
        tickers = stats.index
        mu_arr = stats["mu"].values
        sigma_arr = stats["sigma"].values
        N = len(tickers)
        # 构造 (N,N) 的比较矩阵：行 i 被列 j 支配？
        mu_i = mu_arr.reshape(N, 1)
        mu_j = mu_arr.reshape(1, N)
        sigma_i = sigma_arr.reshape(N, 1)
        sigma_j = sigma_arr.reshape(1, N)
        cond_mu = (mu_j >= mu_i)
        cond_sigma = (sigma_j <= sigma_i)
        strict = (mu_j > mu_i) | (sigma_j < sigma_i)
        dom_mat = cond_mu & cond_sigma & strict
        # 去掉自身比较
        np.fill_diagonal(dom_mat, False)
        dominated_count = dom_mat.sum(axis=1)
        # 严格非支配（Pareto front）：未被任何股票支配
        selected_mask = dominated_count == 0
        selected_idx = tickers[selected_mask]

        # 若需要，按夏普比率补齐到 target_k
        if target_k is not None and len(selected_idx) < target_k:
            # 计算 Sharpe，忽略 σ<=0 的情况
            stats["sharpe"] = stats["mu"] / stats["sigma"].replace(0, np.nan)
            remaining = stats.index.difference(selected_idx)
            remaining = [t for t in remaining if np.isfinite(stats.loc[t, "sharpe"]) ]
            top_fill = (
                stats.loc[remaining]
                .sort_values(by="sharpe", ascending=False)
            )
            need = max(0, int(target_k) - len(selected_idx))
            to_add = list(top_fill.index[:need])
            selected_idx = pd.Index(selected_idx.tolist() + to_add)

        # 绘制 (σ, μ) 散点图：被 ≤2 支配/或 Sharpe 补齐 的为紫色，其余为黑色
        plt.figure(figsize=(7, 5))
        # 先画其他点（黑色）
        others = stats.index.difference(selected_idx)
        plt.scatter(stats.loc[others, "sigma"], stats.loc[others, "mu"],
                    c="#000000", s=18, alpha=0.7, label="Others")
        # 再画所选集合（紫色）
        # 使用 rc_context 禁用 usetex，避免环境依赖导致的渲染报错
        try:
            with plt.rc_context({"text.usetex": False}):
                plt.scatter(stats.loc[selected_idx, "sigma"], stats.loc[selected_idx, "mu"],
                            c="#7E57C2", s=36, alpha=0.95, label="Selected")
                plt.xlabel(r"$\sigma$ (volatility)")
                plt.ylabel(r"$\mu$ (mean return)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(plot_path)
        except Exception:
            # 兜底：若仍失败，降级去掉数学文本渲染
            try:
                with plt.rc_context({"text.usetex": False}):
                    plt.xlabel("sigma (volatility)")
                    plt.ylabel("mu (mean return)")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(plot_path)
            finally:
                pass
        finally:
            plt.close()

        # 构造精选后的收益矩阵（按原列顺序保留）
        selected_cols = [c for c in R.columns if c in set(selected_idx)]
        R_clean = R.loc[:, selected_cols]

        if output_csv:
            R_clean.to_csv(output_csv, index=True)

        return R_clean

    def compute_correlation_matrix(self, R: pd.DataFrame, output_csv: Optional[str] = None) -> pd.DataFrame:
        """
        基于 R_full 按列归一化后计算关联矩阵 C，G[t,i] = (r_{t,i} - <r_i>)/sigma_i，C = (1/T) G^T G。
        """
        T = R.shape[0]
        mu = R.mean(axis=0)
        # 使用总体标准差(ddof=0)，为了让G矩阵中每个元素方差为1
        sigma = R.std(axis=0, ddof=0)
        # 去除标准差为 0 的列，避免除零
        nonzero = sigma != 0
        R = R.loc[:, nonzero]
        mu = mu[nonzero]
        sigma = sigma[nonzero]
        # 标准化后的收益矩阵 G
        G = (R - mu) / sigma
        # 关联矩阵 C = (1/T) G^T G
        C = (G.T @ G) / T
        if output_csv:
            C.to_csv(output_csv, index=True)
        return C

    def compute_covariance_matrix_direct(self, R: pd.DataFrame, output_csv: Optional[str] = None) -> pd.DataFrame:
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

    def spectral_decomposition(self, C: pd.DataFrame):
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

    def denoising(self, C: pd.DataFrame, T: int, N: int, output_csv: Optional[str] = None) -> pd.DataFrame:
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
        vals_desc, vecs_desc = self.spectral_decomposition(C)
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

    def compute_std_matrix(self, R: pd.DataFrame, output_csv: Optional[str] = None) -> pd.DataFrame:
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

    def compute_covariance_matrix(self, C: pd.DataFrame, D: pd.DataFrame, output_csv: Optional[str] = None) -> pd.DataFrame:
        """
        基于关联矩阵 C (可能是去噪声后的) 和标准差对角矩阵 D 计算协方差矩阵 CV，CV = D C D，其中 D 为对角标准差矩阵。
        """
        CV = D @ C @ D
        if output_csv:
            CV.to_csv(output_csv, index=True)
        return CV

    def optimize_portfolio(self, beta: Union[pd.Series, pd.DataFrame, np.ndarray], C: pd.DataFrame):
        """
        根据给定的 beta 向量与协方差矩阵 C，计算最优权重向量 omega，并以与 beta 相同的格式返回。

        数学定义（beta 与 u 视为行向量，C 取逆）：
            uu = u C^{-1} u^T
            ub = u C^{-1} beta^T
            bu = beta C^{-1} u^T
            bb = beta C^{-1} beta^T
            omega = ((bb - ub) u C^{-1} + (uu - bu) beta C^{-1}) / (uu*bb - ub*bu)

        参数：
            beta: 资产的 beta 向量。支持以下格式：
                  - pandas.Series（index 为资产代码）
                  - pandas.DataFrame（1×N 行向量 或 N×1 列向量）
                  - numpy.ndarray（形状 (N,), (1,N) 或 (N,1)）
            C:    协方差矩阵（N×N，行列索引为资产代码）。

        返回：与传入 beta 相同的类型与“行/列”形状。
        """
        # 提取并（必要时）对齐资产顺序
        tickers = list(C.columns)
        X = C.values
        # 逆或伪逆，提升鲁棒性
        try:
            X_inv = np.linalg.inv(X)
        except np.linalg.LinAlgError:
            X_inv = np.linalg.pinv(X)

        # 将 beta 标准化为 1×N 的 numpy 行向量，同时记录其原始“格式”以便还原
        beta_type = None
        beta_meta = {}

        if isinstance(beta, pd.Series):
            beta_type = 'series'
            b_series = beta.reindex(tickers)
            if b_series.isna().any():
                missing = list(b_series[b_series.isna()].index)
                raise ValueError(f"beta 中缺少与协方差矩阵匹配的资产：{missing}")
            b_row = b_series.values.reshape(1, -1)
        elif isinstance(beta, pd.DataFrame):
            if beta.shape[0] == 1:  # 行向量
                beta_type = 'df_row'
                beta_meta['index'] = beta.index
                b_df = beta.loc[:, tickers]
                b_row = b_df.values  # 1×N
            elif beta.shape[1] == 1:  # 列向量
                beta_type = 'df_col'
                beta_meta['columns'] = beta.columns
                b_df = beta.reindex(tickers)
                if b_df.isna().any().any():
                    raise ValueError("beta 列向量与协方差矩阵资产列表不匹配（存在缺失）。")
                b_row = b_df.values.reshape(1, -1)  # 转为 1×N
            else:
                raise ValueError("beta DataFrame 需为 1×N（行）或 N×1（列）形状。")
        elif isinstance(beta, np.ndarray):
            beta_type = 'ndarray'
            if beta.ndim == 1:
                if beta.shape[0] != len(tickers):
                    raise ValueError("beta 一维数组长度需与 C 维度一致。")
                b_row = beta.reshape(1, -1)
                beta_meta['shape'] = (len(tickers),)
            elif beta.ndim == 2 and beta.shape in [(1, len(tickers)), (len(tickers), 1)]:
                if beta.shape[0] == 1:  # 1×N
                    b_row = beta
                else:  # N×1
                    b_row = beta.T
                beta_meta['shape'] = beta.shape
            else:
                raise ValueError("beta ndarray 需为 (N,), (1,N) 或 (N,1) 形状。")
        else:
            raise TypeError("beta 类型不支持，请使用 pandas Series/DataFrame 或 numpy ndarray。")

        # u 行向量
        u_row = np.ones((1, len(tickers)), dtype=float)

        # 先计算 uC^{-1} 与 bC^{-1}
        uX = u_row @ X_inv           # 1×N
        bX = b_row @ X_inv           # 1×N

        # 标量项
        uu = float(uX @ u_row.T)
        ub = float(uX @ b_row.T)
        bu = float(bX @ u_row.T)
        bb = float(bX @ b_row.T)

        denom = uu * bb - ub * bu
        if not np.isfinite(denom) or abs(denom) < 1e-12:
            raise ValueError("目标函数的分母 uu*bb - ub*bu 过小或无效，无法稳定求解权重。")

        omega_row = ((bb - ub) * uX + (uu - bu) * bX) / denom  # 1×N

        # 还原为与 beta 相同的外观/格式
        if beta_type == 'series':
            return pd.Series(omega_row.ravel(), index=tickers, name='omega')
        if beta_type == 'df_row':
            return pd.DataFrame(omega_row, index=beta_meta['index'], columns=tickers)
        if beta_type == 'df_col':
            return pd.DataFrame(omega_row.T, index=tickers, columns=beta_meta['columns'])
        if beta_type == 'ndarray':
            shp = beta_meta.get('shape')
            if shp == (len(tickers),):
                return omega_row.ravel()
            if shp == (1, len(tickers)):
                return omega_row
            if shp == (len(tickers), 1):
                return omega_row.T
        # 理论上不会到达此处
        return pd.Series(omega_row.ravel(), index=tickers, name='omega')

    def compute_beta_factors(self, R_index: pd.DataFrame, R_assets: pd.DataFrame) -> pd.Series:
        """
        计算多只股票相对于“市场指数”的 Beta 因子：
            beta_i = Cov(r_i, r_m) / Var(r_m)

        参数：
            R_index: 指数的收益矩阵（应为单列，索引为日期）
            R_assets: 多只股票的收益矩阵（列为股票，索引为日期）

        处理步骤：
            - 按日期对齐（内连接），并去除对齐后的 NaN
            - 采用总体协方差/方差（均值去除后再按 T 的平均）以与本项目其它计算保持一致

        返回：
            pandas Series，index=R_assets 的列名，值为对应的 beta
        """
        if R_index.shape[1] != 1:
            # 若传入多列，取第一列并提示
            idx_cols = list(R_index.columns)
            R_index = R_index.iloc[:, [0]]
            print(f"警告：R_index 包含多列，已使用第一列 {idx_cols[0]} 进行 Beta 计算。")

        # 对齐日期
        df = R_assets.join(R_index, how="inner")
        df = df.dropna(how="any")
        if df.shape[0] < 2:
            raise ValueError("用于计算 Beta 的对齐样本不足（少于 2 行）。")

        # 拆分指数列与资产列
        m = df.iloc[:, -1]  # 最后一列为指数
        X = df.iloc[:, :-1] # 前面的列为资产

        # 指数的均值与方差（总体）
        mu_m = m.mean()
        var_m = ((m - mu_m) ** 2).mean()
        if var_m == 0 or not np.isfinite(var_m):
            raise ValueError("指数收益的方差为 0 或无效，无法计算 Beta。")

        # 各资产的均值
        mu_i = X.mean(axis=0)
        # 协方差：E[(r_i - mu_i)(r_m - mu_m)]
        cov_im = ((X - mu_i) * (m - mu_m).values.reshape(-1, 1)).mean(axis=0)
        beta = cov_im / var_m
        beta.name = "beta"
        return beta

    def plot_eigensystem(self, eigvals: np.ndarray, eigvecs: np.ndarray, T: int, N: int,
                          eigenvalues_pdf: str = "eigenvalues.pdf", eigenvectors_pdf: str = "eigenvectors.pdf") -> None:
        """
        绘制谱系结果：
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
        plt.hist(eigvals, bins=bins, density=True, alpha=0.7, color="#4C78A8", edgecolor="white", label="Empirical")

        # 理论 MP 密度（带根号的正确公式）
        lam_grid = np.linspace(max(lam_minus, 1e-9), lam_plus, 1000)
        support_mask = (lam_grid >= lam_minus) & (lam_grid <= lam_plus)
        mp_density = np.zeros_like(lam_grid)
        mp_density[support_mask] = (Q / (2 * np.pi)) * (
            np.sqrt((lam_plus - lam_grid[support_mask]) * (lam_grid[support_mask] - lam_minus)) / lam_grid[support_mask]
        )
        plt.plot(lam_grid, mp_density, "-", color="#F58518", linewidth=2.2, label="RMT")

        # 不再绘制竖线；仅在图例中展示 λ± 的数值（用空数据曲线生成图例条目）
        plt.plot([], [], color="#E45756", linestyle="--", linewidth=1.5, label=rf"$\lambda_- = {lam_minus:.3f}$")
        plt.plot([], [], color="#72B7B2", linestyle="--", linewidth=1.5, label=rf"$\lambda_+ = {lam_plus:.3f}$")
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"$p(\lambda)$")
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
                ax.hist(vec_scaled, bins=50, density=True, alpha=0.9, color="#2E7D32", edgecolor="#1B5E20", label="Empirical")
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
