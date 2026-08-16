"""Agent 实时工具：当前时间 / 当前地点 / 当前天气。

这些工具由 Agent 图中的大模型按需调用（ReAct 循环），
与知识库检索工具（定义在 app/graphs.py）共同组成 Agent 的工具集。
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from langchain_core.tools import tool

WMO_WEATHER_CODES: dict[int, str] = {
    0: "晴",
    1: "基本晴朗",
    2: "多云",
    3: "阴天",
    45: "雾",
    48: "雾凇",
    51: "毛毛雨（小）",
    53: "毛毛雨（中）",
    55: "毛毛雨（大）",
    56: "冻毛毛雨（小）",
    57: "冻毛毛雨（大）",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨（小）",
    67: "冻雨（大）",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "阵雨（小）",
    81: "阵雨（中）",
    82: "阵雨（大）",
    85: "阵雪（小）",
    86: "阵雪（大）",
    95: "雷雨",
    96: "雷雨伴冰雹（小）",
    99: "雷雨伴冰雹（大）",
}

# 可通过 .env 固定地点，避免依赖 IP 定位
DEFAULT_LOCATION = os.getenv("DEFAULT_LOCATION", "")
DEFAULT_LATITUDE = os.getenv("DEFAULT_LATITUDE", "")
DEFAULT_LONGITUDE = os.getenv("DEFAULT_LONGITUDE", "")

# 模块级缓存：位置缓存 1 小时，天气缓存 10 分钟
_LOCATION_CACHE: dict[str, Any] = {"ts": 0.0, "data": None}
_WEATHER_CACHE: dict[str, Any] = {"key": None, "ts": 0.0, "data": None}

# 内置中国地级市坐标表（离线兜底，覆盖 356 个城市）
try:
    _CHINA_CITIES: dict[str, list[float]] = json.loads(
        (Path(__file__).resolve().parent / "china_cities.json").read_text(
            encoding="utf-8"
        )
    )
except Exception:
    _CHINA_CITIES = {}


def _local_now() -> datetime:
    """返回北京时间；无法解析时回退到系统本地时间。"""
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("Asia/Shanghai"))
    except Exception:
        return datetime.now()


def _get_current_location() -> dict:
    """获取当前大致地理位置（IP 定位），带 1 小时缓存。"""
    now = time.time()
    if _LOCATION_CACHE["data"] and now - _LOCATION_CACHE["ts"] < 3600:
        return _LOCATION_CACHE["data"]

    # 已配置固定地点时直接使用，跳过网络请求
    if DEFAULT_LATITUDE and DEFAULT_LONGITUDE:
        location = {
            "city": DEFAULT_LOCATION or "配置的地点",
            "region": DEFAULT_LOCATION or "",
            "country": "",
            "lat": float(DEFAULT_LATITUDE),
            "lon": float(DEFAULT_LONGITUDE),
            "timezone": "Asia/Shanghai",
        }
        _LOCATION_CACHE.update(ts=now, data=location)
        return location

    # 依次尝试多个 IP 定位服务，全部失败时返回兜底结果
    for url in (
        "https://ip-api.com/json/?fields=status,country,regionName,city,lat,lon,timezone&lang=zh-CN",
        "https://ipinfo.io/json",
    ):
        try:
            location = _fetch_location_from(url)
            if location.get("lat") is not None:
                _LOCATION_CACHE.update(ts=now, data=location)
                return location
        except Exception:
            continue

    fallback = {
        "city": DEFAULT_LOCATION or "未知",
        "region": "",
        "country": "",
        "lat": None,
        "lon": None,
        "timezone": "Asia/Shanghai",
    }
    _LOCATION_CACHE.update(ts=now, data=fallback)
    return fallback


def _fetch_location_from(url: str) -> dict:
    """调用单个 IP 定位服务，返回统一格式的位置信息（失败返回空 dict）。"""
    data = requests.get(url, timeout=4).json()
    if url.startswith("https://ip-api.com"):
        if data.get("status") != "success":
            return {}
        return {
            "city": data.get("city") or "",
            "region": data.get("regionName") or "",
            "country": data.get("country") or "",
            "lat": data.get("lat"),
            "lon": data.get("lon"),
            "timezone": data.get("timezone") or "Asia/Shanghai",
        }
    loc_parts = (data.get("loc") or "").split(",")
    return {
        "city": data.get("city") or "",
        "region": data.get("region") or "",
        "country": data.get("country") or "",
        "lat": float(loc_parts[0]) if len(loc_parts) > 0 else None,
        "lon": float(loc_parts[1]) if len(loc_parts) > 1 else None,
        "timezone": data.get("timezone") or "Asia/Shanghai",
    }


def _geocode(city: str) -> dict | None:
    """把城市名转成经纬度。

    优先查内置中国城市坐标表（离线、稳定），查不到再回退到
    Open-Meteo 地理编码（覆盖国外城市等场景）。
    """
    normalized = (
        city.strip()
        .replace("市", "")
        .replace("省", "")
        .replace("特别行政区", "")
    )
    coords = _CHINA_CITIES.get(normalized) or _CHINA_CITIES.get(city.strip())
    if coords:
        return {
            "city": normalized,
            "country": "中国",
            "lat": coords[1],
            "lon": coords[0],
        }
    try:
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city.strip(),
                "count": 1,
                "language": "zh",
                "format": "json",
            },
            timeout=5,
        )
        results = (resp.json() or {}).get("results") or []
        if results:
            item = results[0]
            return {
                "city": item.get("name") or city,
                "country": item.get("country") or "",
                "lat": item["latitude"],
                "lon": item["longitude"],
            }
    except Exception:
        return None
    return None


def _fetch_weather(lat: float, lon: float) -> dict | None:
    """从 Open-Meteo 获取实时天气（免费、无需 API Key），带 10 分钟缓存。"""
    cache_key = (round(lat, 2), round(lon, 2))
    now = time.time()
    if _WEATHER_CACHE["key"] == cache_key and now - _WEATHER_CACHE["ts"] < 600:
        return _WEATHER_CACHE["data"]
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "weather_code,wind_speed_10m"
                ),
                "timezone": "auto",
            },
            timeout=6,
        )
        current = (resp.json() or {}).get("current") or {}
        if "temperature_2m" in current:
            _WEATHER_CACHE.update(key=cache_key, ts=now, data=current)
            return current
    except Exception:
        return None
    return None


@tool
def get_current_time() -> str:
    """获取当前的日期和时间（北京时间）。当用户询问“现在几点”“今天是几号”等问题时调用。"""
    now = _local_now()
    weekday = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")[
        now.weekday()
    ]
    return (
        f"现在是 {now.year} 年 {now.month} 月 {now.day} 日 {weekday} "
        f"{now.hour:02d}:{now.minute:02d}（北京时间，UTC+8）。"
    )


@tool
def get_current_location() -> str:
    """获取当前的大致地理位置（基于 IP 定位，返回城市/地区）。当用户询问“我现在在哪里”等问题时调用。"""
    location = _get_current_location()
    parts = [
        part
        for part in (
            location.get("country"),
            location.get("region"),
            location.get("city"),
        )
        if part
    ]
    if not parts:
        return "无法获取当前位置。"
    return (
        f"当前位置：{' '.join(parts)}"
        f"（经度 {location['lon']}，纬度 {location['lat']}）"
    )


@tool
def get_current_weather(location: str = "") -> str:
    """查询指定城市的实时天气。当用户询问天气、气温、会不会下雨等问题时调用；
    用户提到地点时必须把城市名传给 location 参数，不能留空。"""
    if location.strip():
        target = _geocode(location.strip())
        if target is None:
            return (
                f"无法定位城市「{location.strip()}」的坐标，"
                "请让用户提供更大范围的地名（如省份或国家）。"
            )
        city_name = target["city"]
    else:
        current_loc = _get_current_location()
        city_name = "当前位置"
        if current_loc.get("lat") is None:
            return "无法定位查询天气的地点，请直接告诉我要查询的城市名。"
        target = {"lat": current_loc["lat"], "lon": current_loc["lon"]}

    current = _fetch_weather(target["lat"], target["lon"])
    if current is None:
        return f"无法获取 {city_name} 的实时天气（网络或天气服务不可用）。"

    code = current.get("weather_code", 0)
    description = WMO_WEATHER_CODES.get(code, f"未知天气代码 {code}")
    parts = [f"{city_name} 当前天气：{description}"]
    temperature = current.get("temperature_2m")
    if temperature is not None:
        parts.append(f"气温 {temperature}°C")
    apparent = current.get("apparent_temperature")
    if apparent is not None:
        parts.append(f"体感 {apparent}°C")
    humidity = current.get("relative_humidity_2m")
    if humidity is not None:
        parts.append(f"湿度 {humidity}%")
    wind = current.get("wind_speed_10m")
    if wind is not None:
        parts.append(f"风速 {wind} km/h")
    return "，".join(parts) + "。"


AGENT_TOOLS = [get_current_time, get_current_location, get_current_weather]
