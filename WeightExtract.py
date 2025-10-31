import pandas as pd
import re

# 读取特征向量数据
filepath2 = 'eigenvector_2.csv'
filepath3 = 'eigenvector_3.csv'


def PNweight(path, name):
    df = pd.read_csv(path)

    # 按载荷绝对值排序后拆分正负方向，便于后续挑选代表性股票
    positive_stocks = df[df[name] > 0].sort_values(by=name, ascending=False)
    negative_stocks = df[df[name] < 0].sort_values(by=name)

    positive_list = positive_stocks.iloc[:, 0].tolist()
    negative_list = negative_stocks.iloc[:, 0].tolist()
    return [positive_list, negative_list]


[u2p, u2n] = PNweight(filepath2, "u2")
[u3p, u3n] = PNweight(filepath3, "u3")

import baostock as bs
from collections import defaultdict

_bs_logged_in = False


SIMPLIFIED_CODE_MAP = {
    'J66': '银行',
    'J67': '证券',
    'J68': '保险',
    'J69': '多元金融',
    'C39': '电子',
    'C38': '电气设备',
    'C37': '汽车',
    'C36': '机械设备',
    'C30': '有色金属',
    'C31': '有色金属',
    'C32': '有色金属',
    'C33': '汽车',
    'C27': '医药',
    'C26': '化工',
    'C25': '化纤',
    'C22': '造纸',
    'C17': '纺织服装',
    'C23': '印刷包装',
    'C13': '农产品加工',
    'D44': '电力',
    'D45': '燃气',
    'B06': '煤炭',
    'B07': '石油石化',
    'E48': '建筑',
    'F51': '批发零售',
    'G53': '运输',
    'H61': '住宿餐饮',
    'I63': '电信',
    'I64': '互联网',
}


SIMPLIFIED_NAME_MAP = {
    '货币金融服务': '银行',
    '资本市场服务': '证券',
    '银行': '银行',
    '证券': '证券',
    '保险业': '保险',
    '电子信息技术': '电子',
    '电子': '电子',
    '半导体': '半导体',
    '计算机': '计算机',
    '计算机、通信和其他电子设备制造业': '电子',
    '电气机械和器材制造业': '电气设备',
    '医药制造业': '医药',
    '化学原料和化学制品制造业': '化工',
    '汽车制造业': '汽车',
    '电力、热力生产和供应业': '电力',
    '软件和信息技术服务业': '软件',
    '互联网和相关服务': '互联网',
    '房地产业': '房地产',
}


KEYWORD_SIMPLIFICATIONS = [
    ('银行', '银行'),
    ('资本市场', '证券'),
    ('证券', '证券'),
    ('保险', '保险'),
    ('信托', '信托'),
    ('基金', '基金'),
    ('电力', '电力'),
    ('光伏', '光伏'),
    ('新能源', '新能源'),
    ('电气', '电气设备'),
    ('半导体', '半导体'),
    ('集成电路', '半导体'),
    ('芯片', '半导体'),
    ('电子', '电子'),
    ('通信', '通信'),
    ('计算机', '计算机'),
    ('软件', '软件'),
    ('互联网', '互联网'),
    ('医药', '医药'),
    ('生物', '生物医药'),
    ('医疗', '医疗服务'),
    ('化学', '化工'),
    ('化工', '化工'),
    ('化肥', '化工'),
    ('汽车', '汽车'),
    ('家用电器', '家电'),
    ('家电', '家电'),
    ('机械', '机械设备'),
    ('钢铁', '钢铁'),
    ('有色', '有色金属'),
    ('煤炭', '煤炭'),
    ('石油', '石油石化'),
    ('天然气', '石油石化'),
    ('农业', '农业'),
    ('农产品', '农业'),
    ('白酒', '白酒'),
    ('酿酒', '白酒'),
]


def simplify_industry_name(raw_label):
    if raw_label is None:
        return '未知行业'
    text = str(raw_label).strip()
    if not text:
        return '未知行业'

    code_match = re.match(r'^([A-Z]+\d+)', text)
    if code_match:
        prefix = code_match.group(1)
        if prefix in SIMPLIFIED_CODE_MAP:
            return SIMPLIFIED_CODE_MAP[prefix]
    stripped = re.sub(r'^[A-Z]+\d+', '', text).strip()
    candidate = stripped or text

    if candidate in SIMPLIFIED_NAME_MAP:
        return SIMPLIFIED_NAME_MAP[candidate]

    for keyword, label in KEYWORD_SIMPLIFICATIONS:
        if keyword in candidate:
            return label

    for sep in ['、', '-', '—', '及', '和', ' '] :
        if sep in candidate:
            segment = candidate.split(sep)[0].strip()
            if segment:
                return segment

    return candidate


def ensure_bs_login():
    # 确保 Baostock 会话仅登录一次，避免重复握手
    global _bs_logged_in
    if not _bs_logged_in:
        login_info = bs.login()
        if login_info.error_code != '0':
            raise RuntimeError(f"baostock 登录失败: {login_info.error_msg}")
        _bs_logged_in = True
    return _bs_logged_in


def cleanup_bs():
    global _bs_logged_in
    if _bs_logged_in:
        bs.logout()
        _bs_logged_in = False


def get_stock_info(stock_code):
    # 使用 Baostock 获取行业信息
    ensure_bs_login()
    rs = bs.query_stock_industry(code=stock_code)
    if rs.error_code != '0':
        return None

    fields = list(rs.fields) if rs.fields else []
    industry_idx = fields.index('industry') if 'industry' in fields else None
    industry_class_idx = fields.index('industryClassification') if 'industryClassification' in fields else None

    primary_label = None
    secondary_label = None
    while (rs.error_code == '0') & rs.next():
        row = rs.get_row_data()
        if industry_idx is not None and industry_idx < len(row):
            primary_label = row[industry_idx]
        if industry_class_idx is not None and industry_class_idx < len(row):
            secondary_label = row[industry_class_idx]
        if primary_label or secondary_label:
            break
        if row:
            secondary_label = row[-1]
        break
    raw_label = primary_label or secondary_label
    return simplify_industry_name(raw_label)


industry_cache = {}


def get_industry_with_cache(full_code):
    cache_key = full_code.lower()
    if cache_key not in industry_cache:
        try:
            industry_cache[cache_key] = get_stock_info(full_code)
        except Exception:
            industry_cache[cache_key] = None
    cached_value = industry_cache[cache_key]
    return cached_value if cached_value else '未知行业'


def pick_similar_industry(codes, sample_size=10, primary_ratio=0.6, min_industries=3):
    industry_groups = defaultdict(list)
    for code in codes:
        industry = get_industry_with_cache(code)
        industry_groups[industry].append(code)

    # 根据行业样本数量排序，优先选择主导行业
    sorted_groups = sorted(industry_groups.items(), key=lambda x: len(x[1]), reverse=True)

    selected = []
    used_industries = set()

    if sorted_groups:
        target_industry_count = min(len(sorted_groups), max(min_industries, 1))
        primary_industry, primary_codes = sorted_groups[0]
        # 留出位置给其他行业，避免行业过于集中
        max_primary = sample_size - max(target_industry_count - 1, 0)
        base_primary = int(sample_size * primary_ratio)
        primary_target = max(min(base_primary, max_primary), 1)

        selected.extend(primary_codes[:primary_target])
        used_industries.add(primary_industry)

        # 先保证至少有 min_industries 个行业（若数据允许）
        for industry, industry_codes in sorted_groups[1:]:
            if len(selected) >= sample_size:
                break
            if industry in used_industries or not industry_codes:
                continue
            if len(used_industries) < target_industry_count:
                selected.append(industry_codes[0])
                used_industries.add(industry)

        # 再按剩余容量补足股票，保持行业多样性
        remaining_pool = []
        for industry, industry_codes in sorted_groups:
            for code in industry_codes:
                if code not in selected:
                    remaining_pool.append(code)

        for code in remaining_pool:
            if len(selected) >= sample_size:
                break
            selected.append(code)

    # 若仍不足，直接补齐
    if len(selected) < sample_size:
        for code in codes:
            if code not in selected:
                selected.append(code)
            if len(selected) >= sample_size:
                break

    return selected[:sample_size]


u2p_selected = pick_similar_industry(u2p)
u2n_selected = pick_similar_industry(u2n)
u3p_selected = pick_similar_industry(u3p)
u3n_selected = pick_similar_industry(u3n)

u2p_industry = [get_industry_with_cache(code) for code in u2p_selected]
u2n_industry = [get_industry_with_cache(code) for code in u2n_selected]
u3p_industry = [get_industry_with_cache(code) for code in u3p_selected]
u3n_industry = [get_industry_with_cache(code) for code in u3n_selected]


# 创建DataFrame
df = pd.DataFrame({
    '$u_{2,+}$ (编号)': u2p_selected,
    '$u_{2,+}$ (行业)': u2p_industry,
    '$u_{2,-}$ (编号)': u2n_selected,
    '$u_{2,-}$ (行业)': u2n_industry,
    '$u_{3,+}$ (编号)': u3p_selected,
    '$u_{3,+}$ (行业)': u3p_industry,
    '$u_{3,-}$ (编号)': u3n_selected,
    '$u_{3,-}$ (行业)': u3n_industry
})

def write_markdown_table(dataframe, path):
    # 将 DataFrame 导出为 Markdown 表格以便直接用于报告
    headers = dataframe.columns.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in dataframe.iterrows():
        values = ["" if pd.isna(val) else str(val).replace("|", "\\|") for val in row.tolist()]
        lines.append("| " + " | ".join(values) + " |")
    with open(path, 'w', encoding='utf-8') as md_file:
        md_file.write("\n".join(lines))


markdown_path = 'EigenvectorSampleTable.md'
write_markdown_table(df, markdown_path)
print(f"数据已导出为 Markdown 表格: {markdown_path}")

cleanup_bs()


