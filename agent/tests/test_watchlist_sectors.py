"""Sector view: Shenwan level-1 map, per-sector top-N ranking, and its route.

The ranking is read-only over stored ``scan_scores`` rows, so every test runs
a real ``WatchlistStore`` on a tmp_path database (no network, no scan).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api import watchlist_routes  # noqa: E402
from src.market_scanner import sectors  # noqa: E402
from src.market_scanner.sectors import (  # noqa: E402
    OTHER_SECTOR,
    SW_L1,
    is_known_industry,
    sector_of,
)
from src.watchlist.service import WatchlistService  # noqa: E402
from src.watchlist.store import WatchlistStore  # noqa: E402

#: Every industry label stored by the live scans up to 2026-10-21 (128).
LIVE_LABELS = (
    "IT服务Ⅱ 一般零售 专业工程 专业服务 专业连锁Ⅱ 专用设备 个护用品 中药Ⅱ 乘用车 "
    "互联网电商 休闲食品 体育Ⅱ 保险Ⅱ 元件 光伏设备 光学光电子 其他家电Ⅱ 其他电子Ⅱ "
    "其他电源设备Ⅱ 养殖业 军工电子Ⅱ 农业综合Ⅱ 农产品加工 农化制品 冶钢原料 出版 "
    "动物保健Ⅱ 包装印刷 化妆品 化学制品 化学制药 化学原料 化学纤维 医疗器械 医疗服务 "
    "医疗美容 医药商业 半导体 厨卫电器 商用车 地面兵装Ⅱ 基础建设 塑料 多元金融 家居用品 "
    "家电零部件Ⅱ 小家电 小金属 工业金属 工程咨询服务Ⅱ 工程机械 广告营销 影视院线 "
    "房地产开发 房地产服务 房屋建设Ⅱ 摩托车及其他 教育 数字媒体 文娱用品 旅游及景区 "
    "旅游零售Ⅱ 普钢 服装家纺 林业Ⅱ 橡胶 水泥 汽车服务 汽车零部件 油服工程 油气开采Ⅱ "
    "消费电子 渔业 游戏Ⅱ 炼化及贸易 焦炭Ⅱ 煤炭开采 照明设备Ⅱ 燃气Ⅱ 物流 特钢Ⅱ "
    "环保设备Ⅱ 环境治理 玻璃玻纤 生物制品 电力 电子化学品Ⅱ 电机Ⅱ 电池 电网设备 "
    "电视广播Ⅱ 白色家电 白酒Ⅱ 种植业 纺织制造 综合Ⅱ 能源金属 自动化设备 航天装备Ⅱ "
    "航海装备Ⅱ 航空机场 航空装备Ⅱ 航运港口 装修建材 装修装饰Ⅱ 计算机设备 证券Ⅱ "
    "调味发酵品Ⅱ 贵金属 贸易Ⅱ 轨交设备Ⅱ 软件开发 通信服务 通信设备 通用设备 造纸 "
    "酒店餐饮 金属新材料 铁路公路 银行Ⅱ 非白酒 非金属材料Ⅱ 风电设备 食品加工 饮料乳品 "
    "饰品 饲料 黑色家电"
).split()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _score(code, industry, score, *, name=None, board="main_sh", excluded=None):
    if excluded:
        return {
            "code": code, "name": name or code, "board": board, "industry": None,
            "score": None, "fundamental": None, "trend": None, "flow": None,
            "exclusion_reason": excluded,
        }
    return {
        "code": code, "name": name or code, "board": board, "industry": industry,
        "score": score, "fundamental": score - 1, "trend": score - 2,
        "flow": score - 3, "exclusion_reason": None,
    }


def _save(store, run_id, scores, *, picks=(), status="partial", trade_date="2026-10-21"):
    run = {
        "run_id": run_id,
        "trade_date": trade_date,
        "started_at": f"{trade_date}T18:00:00+08:00",
        "finished_at": f"{trade_date}T18:20:00+08:00",
        "status": status,
        "trigger": "schedule",
        "ai_status": "reviewed",
        "scores": scores,
    }
    by_code = {s["code"]: s for s in scores}
    new = [
        (code, rank, by_code[code]["score"], {"code": code, "rule_rank": rank})
        for rank, code in enumerate(picks, start=1)
    ]
    store.save_run(run, changes={"new": new, "kept": [], "dropped": []})


@pytest.fixture()
def store(tmp_path):
    return WatchlistStore(path=tmp_path / "watchlist.sqlite3")


@pytest.fixture()
def service(store):
    return WatchlistService(store)


# ---------------------------------------------------------------------------
# the Shenwan map
# ---------------------------------------------------------------------------


def test_map_has_31_sectors_and_131_industries_with_real_parents():
    assert len(SW_L1) == 31 and len(set(SW_L1)) == 31
    table = sectors._SW_L2_TO_L1
    assert len(table) == 131
    assert set(table.values()) == set(SW_L1)  # every sector has an industry


def test_every_stored_live_label_maps_to_a_real_sector():
    assert len(LIVE_LABELS) == 128
    unmapped = [label for label in LIVE_LABELS if not is_known_industry(label)]
    assert unmapped == []


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("半导体", "电子"),
        ("白酒Ⅱ", "食品饮料"),
        ("IT服务Ⅱ", "计算机"),
        ("银行Ⅱ", "银行"),  # pre-split Eastmoney label (alias)
        ("城商行Ⅱ", "银行"),  # official split label
        ("  化学制药 ", "医药生物"),
        ("未分类", OTHER_SECTOR),
        ("某个新行业", OTHER_SECTOR),
        ("", OTHER_SECTOR),
        (None, OTHER_SECTOR),
    ],
)
def test_sector_of(label, expected):
    assert sector_of(label) == expected


# ---------------------------------------------------------------------------
# the ranking
# ---------------------------------------------------------------------------


def _mixed_run(store, watching=()):
    scores = [
        _score("600001", "半导体", 90.0),
        _score("600002", "白酒Ⅱ", 88.0),
        _score("600003", "消费电子", 86.0),
        _score("600004", "银行Ⅱ", 70.0),
        _score("600005", "半导体", 60.0),
        _score("600006", "某个新行业", 99.0),
        _score("600007", None, 0, excluded="excluded:st: ST"),
    ]
    _save(store, "run-a", scores, picks=("600001", "600002"))
    for code in watching:
        store.upsert_item(code, None, "manual", "active", "2026-10-21T19:00:00+08:00")


def test_level1_groups_ranks_and_flags(store, service):
    _mixed_run(store, watching=("600003",))
    view = service.get_sector_ranking()

    assert view["run"]["run_id"] == "run-a"
    assert view["run"]["trade_date"] == "2026-10-21"
    assert view["level"] == "l1" and view["limit"] == 10
    assert view["scored_total"] == 6  # the excluded row never counts
    assert view["picks_count"] == 2

    names = [g["name"] for g in view["sectors"]]
    # strongest leader first; the catch-all bucket last despite its 99.0
    assert names == ["电子", "食品饮料", "银行", OTHER_SECTOR]

    electronics = view["sectors"][0]
    assert electronics["count"] == 3
    assert electronics["leader_score"] == 90.0
    rows = [
        (s["code"], s["industry"], s["sector_rank"], s["market_rank"], s["in_picks"], s["watching"])
        for s in electronics["top"]
    ]
    # market rank counts the 99.0 "other" stock first
    assert rows == [
        ("600001", "半导体", 1, 2, True, False),
        ("600003", "消费电子", 2, 4, False, True),
        ("600005", "半导体", 3, 6, False, False),
    ]
    assert electronics["top"][0]["sub_scores"] == {
        "fundamental": 89.0, "trend": 88.0, "flow": 87.0,
    }
    assert view["sectors"][2]["top"][0]["sector"] == "银行"
    every_code = {s["code"] for g in view["sectors"] for s in g["top"]}
    assert "600007" not in every_code


def test_level2_groups_by_industry_and_keeps_the_sector(store, service):
    _mixed_run(store)
    view = service.get_sector_ranking(level="l2")
    names = [g["name"] for g in view["sectors"]]
    # an unmapped label is still a real industry at level 2: it ranks by its
    # leader like any other (only its level-1 sector is the catch-all)
    assert names == ["某个新行业", "半导体", "白酒Ⅱ", "消费电子", "银行Ⅱ"]
    assert view["sectors"][0]["top"][0]["sector"] == OTHER_SECTOR
    semis = view["sectors"][1]
    assert [s["code"] for s in semis["top"]] == ["600001", "600005"]
    assert all(s["sector"] == "电子" for s in semis["top"])


def test_level2_unclassified_bucket_sorts_last(store, service):
    _save(
        store,
        "run-a",
        [_score("600001", "未分类", 99.0), _score("600002", "半导体", 50.0)],
    )
    names = [g["name"] for g in service.get_sector_ranking(level="l2")["sectors"]]
    assert names == ["半导体", "未分类"]


def test_limit_cuts_the_list_but_count_stays_full(store, service):
    scores = [_score(f"6000{i:02d}", "半导体", 50.0 + i) for i in range(15)]
    _save(store, "run-a", scores)
    view = service.get_sector_ranking(limit=10)
    (semis,) = view["sectors"]
    assert semis["count"] == 15
    assert len(semis["top"]) == 10
    assert [s["sector_rank"] for s in semis["top"]] == list(range(1, 11))
    assert semis["top"][0]["code"] == "600014"


def test_ties_break_on_code_like_the_pipeline(store, service):
    scores = [
        _score("600009", "半导体", 80.0),
        _score("600001", "半导体", 80.0),
        _score("600005", "白酒Ⅱ", 80.0),
    ]
    _save(store, "run-a", scores)
    view = service.get_sector_ranking()
    flat = sorted(
        ((s["market_rank"], s["code"]) for g in view["sectors"] for s in g["top"])
    )
    assert flat == [(1, "600001"), (2, "600005"), (3, "600009")]


def test_default_run_is_the_latest_successful_one(store, service):
    _save(store, "run-ok", [_score("600001", "半导体", 80.0)])
    _save(
        store, "run-failed", [_score("600002", "白酒Ⅱ", 70.0)],
        status="failed", trade_date="2026-10-22",
    )
    assert service.get_sector_ranking()["run"]["run_id"] == "run-ok"
    # an explicit run id is honoured, whatever its status
    view = service.get_sector_ranking("run-failed")
    assert view["run"]["status"] == "failed"
    assert view["sectors"][0]["top"][0]["code"] == "600002"


def test_unknown_run_is_none_and_empty_store_is_an_empty_view(service):
    assert service.get_sector_ranking("nope") is None
    view = service.get_sector_ranking()
    assert view["run"] is None
    assert view["sectors"] == [] and view["scored_total"] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"level": "l3"}, "unknown sector level"),
        ({"limit": 0}, "sector limit"),
        ({"limit": 51}, "sector limit"),
        ({"limit": True}, "sector limit"),
    ],
)
def test_bad_arguments_raise(service, kwargs, message):
    with pytest.raises(ValueError, match=message):
        service.get_sector_ranking(**kwargs)


def test_ranking_never_writes(store, service):
    _mixed_run(store)
    before = store.path.read_bytes()
    service.get_sector_ranking()
    service.get_sector_ranking(level="l2", limit=50)
    assert store.path.read_bytes() == before


# ---------------------------------------------------------------------------
# the route
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch, store, service):
    # A non-None factory keeps register_watchlist_routes from building the
    # real process scheduler; no `with` block, so no lifespan / server lock.
    monkeypatch.setattr(watchlist_routes, "_SCHEDULER_FACTORY", lambda: None)
    monkeypatch.setattr(watchlist_routes, "_SCHEDULER", None)
    monkeypatch.setattr(watchlist_routes, "_service", lambda: service)
    app = FastAPI()
    watchlist_routes.register_watchlist_routes(app)
    return TestClient(app)


def test_route_returns_the_ranking(client, store):
    _mixed_run(store)
    r = client.get("/api/watchlist/sectors")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    ranking = body["ranking"]
    assert ranking["level"] == "l1" and ranking["limit"] == 10
    assert ranking["sectors"][0]["name"] == "电子"

    r = client.get("/api/watchlist/sectors", params={"level": "l2", "limit": 1, "run_id": "run-a"})
    assert r.status_code == 200
    groups = {g["name"]: g for g in r.json()["ranking"]["sectors"]}
    assert groups["半导体"]["count"] == 2 and len(groups["半导体"]["top"]) == 1


def test_route_unknown_run_404(client, store):
    _mixed_run(store)
    r = client.get("/api/watchlist/sectors", params={"run_id": "nope"})
    assert r.status_code == 404


@pytest.mark.parametrize(
    "params",
    [{"level": "l3"}, {"limit": 0}, {"limit": 51}, {"limit": "x"}],
)
def test_route_rejects_bad_query(client, params):
    assert client.get("/api/watchlist/sectors", params=params).status_code == 422


def test_route_empty_store(client):
    r = client.get("/api/watchlist/sectors")
    assert r.status_code == 200
    assert r.json()["ranking"]["run"] is None


def test_api_server_registers_the_sectors_route():
    import api_server

    paths = {getattr(r, "path", None) for r in api_server.app.routes}
    assert "/api/watchlist/sectors" in paths
