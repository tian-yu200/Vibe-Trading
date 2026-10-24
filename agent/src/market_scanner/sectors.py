"""Shenwan (申万) 2021 sector map: level-2 industry -> level-1 sector.

The scanner's per-stock ``industry`` label is Eastmoney's ``所处行业``
(fundamentals feed), which uses the Shenwan 2021 level-2 names such as
``半导体``, ``白酒Ⅱ`` or ``IT服务Ⅱ``. The watchlist's sector view groups
those labels into the 31 Shenwan level-1 sectors.

Source of the tables below: akshare ``sw_index_first_info`` /
``sw_index_second_info`` (Shenwan official index list), fetched 2026-10-21:
31 level-1 sectors, 131 level-2 industries, each with its parent sector.
Every label seen in the stored scans maps through this table except
``银行Ⅱ``: Eastmoney still reports the banks as one pre-split group, which
Shenwan now splits into four (国有大型银行Ⅱ / 股份制银行Ⅱ / 城商行Ⅱ /
农商行Ⅱ), hence the alias.

Pure data + lookups: no network, no I/O.
"""

from __future__ import annotations

#: The 31 Shenwan 2021 level-1 sectors, in Shenwan code order (801010 ... 801980).
SW_L1: tuple[str, ...] = (
    "农林牧渔",  # 801010.SI
    "基础化工",  # 801030.SI
    "钢铁",  # 801040.SI
    "有色金属",  # 801050.SI
    "电子",  # 801080.SI
    "家用电器",  # 801110.SI
    "食品饮料",  # 801120.SI
    "纺织服饰",  # 801130.SI
    "轻工制造",  # 801140.SI
    "医药生物",  # 801150.SI
    "公用事业",  # 801160.SI
    "交通运输",  # 801170.SI
    "房地产",  # 801180.SI
    "商贸零售",  # 801200.SI
    "社会服务",  # 801210.SI
    "综合",  # 801230.SI
    "建筑材料",  # 801710.SI
    "建筑装饰",  # 801720.SI
    "电力设备",  # 801730.SI
    "国防军工",  # 801740.SI
    "计算机",  # 801750.SI
    "传媒",  # 801760.SI
    "通信",  # 801770.SI
    "银行",  # 801780.SI
    "非银金融",  # 801790.SI
    "汽车",  # 801880.SI
    "机械设备",  # 801890.SI
    "煤炭",  # 801950.SI
    "石油石化",  # 801960.SI
    "环保",  # 801970.SI
    "美容护理",  # 801980.SI
)

#: Shenwan 2021 level-2 industry -> level-1 sector (131 official entries).
_SW_L2_TO_L1: dict[str, str] = {
    # 农林牧渔
    "林业Ⅱ": "农林牧渔",
    "农产品加工": "农林牧渔",
    "饲料": "农林牧渔",
    "渔业": "农林牧渔",
    "种植业": "农林牧渔",
    "养殖业": "农林牧渔",
    "动物保健Ⅱ": "农林牧渔",
    "农业综合Ⅱ": "农林牧渔",
    # 基础化工
    "化学纤维": "基础化工",
    "化学原料": "基础化工",
    "化学制品": "基础化工",
    "塑料": "基础化工",
    "橡胶": "基础化工",
    "农化制品": "基础化工",
    "非金属材料Ⅱ": "基础化工",
    # 钢铁
    "冶钢原料": "钢铁",
    "普钢": "钢铁",
    "特钢Ⅱ": "钢铁",
    # 有色金属
    "金属新材料": "有色金属",
    "贵金属": "有色金属",
    "小金属": "有色金属",
    "工业金属": "有色金属",
    "能源金属": "有色金属",
    # 电子
    "半导体": "电子",
    "其他电子Ⅱ": "电子",
    "元件": "电子",
    "光学光电子": "电子",
    "消费电子": "电子",
    "电子化学品Ⅱ": "电子",
    # 家用电器
    "白色家电": "家用电器",
    "黑色家电": "家用电器",
    "小家电": "家用电器",
    "厨卫电器": "家用电器",
    "照明设备Ⅱ": "家用电器",
    "家电零部件Ⅱ": "家用电器",
    "其他家电Ⅱ": "家用电器",
    # 食品饮料
    "食品加工": "食品饮料",
    "白酒Ⅱ": "食品饮料",
    "非白酒": "食品饮料",
    "饮料乳品": "食品饮料",
    "休闲食品": "食品饮料",
    "调味发酵品Ⅱ": "食品饮料",
    # 纺织服饰
    "纺织制造": "纺织服饰",
    "服装家纺": "纺织服饰",
    "饰品": "纺织服饰",
    # 轻工制造
    "包装印刷": "轻工制造",
    "家居用品": "轻工制造",
    "造纸": "轻工制造",
    "文娱用品": "轻工制造",
    # 医药生物
    "化学制药": "医药生物",
    "生物制品": "医药生物",
    "医疗器械": "医药生物",
    "医药商业": "医药生物",
    "中药Ⅱ": "医药生物",
    "医疗服务": "医药生物",
    # 公用事业
    "电力": "公用事业",
    "燃气Ⅱ": "公用事业",
    # 交通运输
    "物流": "交通运输",
    "铁路公路": "交通运输",
    "航空机场": "交通运输",
    "航运港口": "交通运输",
    # 房地产
    "房地产开发": "房地产",
    "房地产服务": "房地产",
    # 商贸零售
    "贸易Ⅱ": "商贸零售",
    "一般零售": "商贸零售",
    "专业连锁Ⅱ": "商贸零售",
    "互联网电商": "商贸零售",
    "旅游零售Ⅱ": "商贸零售",
    # 社会服务
    "体育Ⅱ": "社会服务",
    "专业服务": "社会服务",
    "酒店餐饮": "社会服务",
    "旅游及景区": "社会服务",
    "教育": "社会服务",
    # 综合
    "综合Ⅱ": "综合",
    # 建筑材料
    "水泥": "建筑材料",
    "玻璃玻纤": "建筑材料",
    "装修建材": "建筑材料",
    # 建筑装饰
    "房屋建设Ⅱ": "建筑装饰",
    "装修装饰Ⅱ": "建筑装饰",
    "基础建设": "建筑装饰",
    "专业工程": "建筑装饰",
    "工程咨询服务Ⅱ": "建筑装饰",
    # 电力设备
    "电机Ⅱ": "电力设备",
    "其他电源设备Ⅱ": "电力设备",
    "光伏设备": "电力设备",
    "风电设备": "电力设备",
    "电池": "电力设备",
    "电网设备": "电力设备",
    # 国防军工
    "航天装备Ⅱ": "国防军工",
    "航空装备Ⅱ": "国防军工",
    "地面兵装Ⅱ": "国防军工",
    "航海装备Ⅱ": "国防军工",
    "军工电子Ⅱ": "国防军工",
    # 计算机
    "计算机设备": "计算机",
    "IT服务Ⅱ": "计算机",
    "软件开发": "计算机",
    # 传媒
    "游戏Ⅱ": "传媒",
    "广告营销": "传媒",
    "影视院线": "传媒",
    "数字媒体": "传媒",
    "出版": "传媒",
    "电视广播Ⅱ": "传媒",
    # 通信
    "通信设备": "通信",
    "通信服务": "通信",
    # 银行
    "国有大型银行Ⅱ": "银行",
    "股份制银行Ⅱ": "银行",
    "城商行Ⅱ": "银行",
    "农商行Ⅱ": "银行",
    # 非银金融
    "多元金融": "非银金融",
    "证券Ⅱ": "非银金融",
    "保险Ⅱ": "非银金融",
    # 汽车
    "汽车服务": "汽车",
    "汽车零部件": "汽车",
    "乘用车": "汽车",
    "商用车": "汽车",
    "摩托车及其他": "汽车",
    # 机械设备
    "通用设备": "机械设备",
    "专用设备": "机械设备",
    "轨交设备Ⅱ": "机械设备",
    "工程机械": "机械设备",
    "自动化设备": "机械设备",
    # 煤炭
    "煤炭开采": "煤炭",
    "焦炭Ⅱ": "煤炭",
    # 石油石化
    "油气开采Ⅱ": "石油石化",
    "油服工程": "石油石化",
    "炼化及贸易": "石油石化",
    # 环保
    "环境治理": "环保",
    "环保设备Ⅱ": "环保",
    # 美容护理
    "个护用品": "美容护理",
    "化妆品": "美容护理",
    "医疗美容": "美容护理",
}

#: Labels the data source uses that are not official level-2 names.
_ALIASES: dict[str, str] = {
    "银行Ⅱ": "银行",
}

#: Bucket for a label outside the table (or a stock with no label). Never
#: silently dropped: the stock stays visible under this name.
OTHER_SECTOR = "其他"

#: The scanner's own placeholder for a stock without an industry label
#: (``pipeline._industry_of``).
UNCLASSIFIED_INDUSTRY = "未分类"


def sector_of(industry: str | None) -> str:
    """Level-1 sector of a level-2 industry label.

    Args:
        industry: The scanner's ``industry`` label (Shenwan level-2 name).

    Returns:
        The Shenwan level-1 sector name, or ``OTHER_SECTOR`` when the label
        is missing or not in the table.
    """
    if not industry:
        return OTHER_SECTOR
    label = str(industry).strip()
    return _SW_L2_TO_L1.get(label) or _ALIASES.get(label) or OTHER_SECTOR


def is_known_industry(industry: str | None) -> bool:
    """True when ``industry`` maps to a real level-1 sector."""
    return sector_of(industry) != OTHER_SECTOR
