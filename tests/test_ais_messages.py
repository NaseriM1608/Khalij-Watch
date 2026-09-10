"""Tests for AIS validation and aisstream parsing.

The emphasis is on dirty input. The ITU sentinels matter most: they are valid
transmissions meaning "not available", and storing them as real measurements
would silently corrupt heading and speed comparisons during SAR fusion.
"""

from __future__ import annotations

import datetime as dt

import pytest

from core.ais.aisstream import (
    SUBSCRIBED_TYPES,
    AisParseError,
    build_subscription,
    parse_go_timestamp,
    parse_position,
    parse_static,
)
from core.ais.messages import (
    AisPositionReport,
    clean_text,
    coarse_class,
    dimensions_to_size,
    is_valid_mmsi,
    normalise_cog,
    normalise_heading,
    normalise_latitude,
    normalise_sog,
)

IRAN_MMSI = 422123456
NOW = dt.datetime(2026, 9, 10, 12, 0, tzinfo=dt.UTC)


def position_envelope(**overrides):
    body = {
        "UserID": IRAN_MMSI,
        "Latitude": 26.5,
        "Longitude": 56.5,
        "Sog": 12.3,
        "Cog": 123.4,
        "TrueHeading": 120,
        "NavigationalStatus": 0,
        "MessageID": 1,
    }
    body.update(overrides)
    return {
        "MessageType": "PositionReport",
        "MetaData": {
            "MMSI": IRAN_MMSI,
            "ShipName": "TEST SHIP @@",
            "time_utc": "2026-09-10 11:59:30.123456789 +0000 UTC",
        },
        "Message": {"PositionReport": body},
    }


# --------------------------------------------------------------------------
# MMSI
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mmsi", [IRAN_MMSI, 470123456, 201000001])
def test_valid_mmsis_are_accepted(mmsi):
    assert is_valid_mmsi(mmsi)


@pytest.mark.parametrize(
    "mmsi",
    [
        0,  # the classic junk value
        123,  # too short
        1_234_567_890,  # too long
        99_999_999,  # 8 digits
        970_123_456,  # MID 970: AIS-SART, not a vessel
        "422123456",  # string, not int
        None,
        True,  # bool is an int subclass; must still be rejected
    ],
)
def test_invalid_mmsis_are_rejected(mmsi):
    assert not is_valid_mmsi(mmsi)


def test_gulf_mid_is_exposed_as_a_flag_hint():
    report = AisPositionReport(mmsi=IRAN_MMSI, ts=NOW, lon=56.5, lat=26.5, source="test")

    assert report.mid == 422
    assert report.flag_hint == "IRN"


def test_non_gulf_mmsi_has_no_flag_hint():
    report = AisPositionReport(mmsi=232123456, ts=NOW, lon=0.0, lat=51.0, source="test")

    assert report.flag_hint is None


# --------------------------------------------------------------------------
# ITU sentinels
# --------------------------------------------------------------------------


def test_latitude_sentinel_becomes_none():
    """91.0 means 'not available', not a position near the pole."""
    assert normalise_latitude(91.0) is None
    assert normalise_latitude(26.5) == 26.5


def test_speed_sentinel_becomes_none():
    assert normalise_sog(102.3) is None
    assert normalise_sog(12.3) == 12.3


def test_course_sentinel_becomes_none():
    """360.0 must not be stored: it would read as due north."""
    assert normalise_cog(360.0) is None
    assert normalise_cog(0.0) == 0.0
    assert normalise_cog(359.9) == 359.9


def test_heading_sentinel_becomes_none():
    assert normalise_heading(511) is None
    assert normalise_heading(120) == 120.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_rejected(value):
    assert normalise_latitude(value) is None
    assert normalise_sog(value) is None


def test_implausible_speed_is_rejected():
    """No surface vessel does 500 knots; that is a decode error or a spoof."""
    assert normalise_sog(500.0) is None


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------


def test_clean_text_strips_ais_padding_and_nul_bytes():
    assert clean_text("EVER GIVEN@@@@") == "EVER GIVEN"
    assert clean_text("BAD\x00NAME") == "BADNAME"
    assert clean_text("@@@@") is None
    assert clean_text("   ") is None
    assert clean_text(None) is None


# --------------------------------------------------------------------------
# Coarse classes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ship_type", "expected"),
    [
        (70, "cargo_tanker"),
        (79, "cargo_tanker"),
        (80, "cargo_tanker"),  # tanker merges with cargo by design
        (89, "cargo_tanker"),
        (30, "fishing"),
        (31, "service_tug"),
        (52, "service_tug"),
        (60, "passenger_other"),
        (37, "passenger_other"),
        (None, None),
        (999, None),
    ],
)
def test_coarse_class_mapping(ship_type, expected):
    assert coarse_class(ship_type) == expected


def test_tanker_and_cargo_are_deliberately_not_separated():
    """They are geometrically near-identical at 10 m; separating them is noise."""
    assert coarse_class(70) == coarse_class(80)


# --------------------------------------------------------------------------
# Dimensions
# --------------------------------------------------------------------------


def test_dimensions_sum_to_length_and_beam():
    assert dimensions_to_size(200, 100, 20, 25) == (300.0, 45.0)


def test_all_zero_dimensions_mean_not_reported():
    assert dimensions_to_size(0, 0, 0, 0) == (None, None)


def test_negative_dimensions_are_rejected():
    assert dimensions_to_size(-1, 100, 20, 25) == (None, None)


# --------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------


def test_go_timestamp_with_nanoseconds_is_parsed():
    parsed = parse_go_timestamp("2026-09-10 11:59:30.123456789 +0000 UTC")

    assert parsed == dt.datetime(2026, 9, 10, 11, 59, 30, 123456, tzinfo=dt.UTC)


def test_go_timestamp_without_fraction_is_parsed():
    assert parse_go_timestamp("2026-09-10 11:59:30 +0000 UTC") == dt.datetime(
        2026, 9, 10, 11, 59, 30, tzinfo=dt.UTC
    )


@pytest.mark.parametrize("raw", ["", "garbage", None, "2026-13-45 99:99:99 +0000 UTC"])
def test_unparseable_timestamps_return_none(raw):
    assert parse_go_timestamp(raw) is None


def test_future_timestamp_is_replaced_with_now():
    envelope = position_envelope()
    envelope["MetaData"]["time_utc"] = "2099-01-01 00:00:00 +0000 UTC"

    report = parse_position(envelope, source="test", now=NOW)

    assert report.ts == NOW


def test_reported_timestamp_is_preferred_when_sane():
    report = parse_position(position_envelope(), source="test", now=NOW)

    assert report.ts == dt.datetime(2026, 9, 10, 11, 59, 30, 123456, tzinfo=dt.UTC)


# --------------------------------------------------------------------------
# Position parsing
# --------------------------------------------------------------------------


def test_parse_position_reads_a_well_formed_message():
    report = parse_position(position_envelope(), source="aisstream", now=NOW)

    assert report.mmsi == IRAN_MMSI
    assert (report.lon, report.lat) == (56.5, 26.5)
    assert report.sog_kn == 12.3
    assert report.heading_deg == 120.0
    assert report.source == "aisstream"


def test_parse_position_nulls_sentinels_rather_than_storing_them():
    envelope = position_envelope(Sog=102.3, Cog=360.0, TrueHeading=511)

    report = parse_position(envelope, source="test", now=NOW)

    assert report.sog_kn is None
    assert report.cog_deg is None
    assert report.heading_deg is None


def test_parse_position_falls_back_to_metadata_coordinates():
    envelope = position_envelope(Latitude=91.0, Longitude=181.0)
    envelope["MetaData"]["latitude"] = 26.6
    envelope["MetaData"]["longitude"] = 56.6

    report = parse_position(envelope, source="test", now=NOW)

    assert (report.lon, report.lat) == (56.6, 26.6)


def test_parse_position_rejects_message_with_no_usable_position():
    envelope = position_envelope(Latitude=91.0, Longitude=181.0)

    with pytest.raises(AisParseError, match="no usable position"):
        parse_position(envelope, source="test", now=NOW)


def test_parse_position_rejects_invalid_mmsi():
    envelope = position_envelope(UserID=0)
    envelope["MetaData"]["MMSI"] = 0

    with pytest.raises(AisParseError, match="no valid MMSI"):
        parse_position(envelope, source="test", now=NOW)


def test_parse_position_rejects_wrong_message_type():
    envelope = position_envelope()
    envelope["MessageType"] = "ShipStaticData"

    with pytest.raises(AisParseError, match="not a position report"):
        parse_position(envelope, source="test", now=NOW)


def test_source_is_mandatory_for_attribution():
    """aiscast re-serves per-source licences, so source must never be blank."""
    with pytest.raises(ValueError, match="source is required"):
        AisPositionReport(mmsi=IRAN_MMSI, ts=NOW, lon=56.5, lat=26.5, source="")


def test_naive_timestamps_are_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        AisPositionReport(
            mmsi=IRAN_MMSI,
            ts=dt.datetime(2026, 9, 10, 12, 0),
            lon=56.5,
            lat=26.5,
            source="test",
        )


# --------------------------------------------------------------------------
# Static data
# --------------------------------------------------------------------------


def test_parse_static_reads_identity_and_dimensions():
    envelope = {
        "MessageType": "ShipStaticData",
        "MetaData": {"MMSI": IRAN_MMSI},
        "Message": {
            "ShipStaticData": {
                "UserID": IRAN_MMSI,
                "Name": "EVER GIVEN@@@",
                "CallSign": "H3RC",
                "ImoNumber": 9811000,
                "Type": 70,
                "Dimension": {"A": 200, "B": 100, "C": 20, "D": 25},
                "Destination": "JEBEL ALI@@",
                "MaximumStaticDraught": 14.5,
            }
        },
    }

    static = parse_static(envelope)

    assert static.name == "EVER GIVEN"
    assert static.imo == 9811000
    assert static.length_m == 300.0
    assert static.beam_m == 45.0
    assert static.destination == "JEBEL ALI"
    assert static.coarse_class == "cargo_tanker"


def test_parse_static_rejects_placeholder_imo():
    """The feed sends IMO 0 constantly; it must not be stored as an identity."""
    envelope = {
        "MessageType": "ShipStaticData",
        "MetaData": {"MMSI": IRAN_MMSI},
        "Message": {"ShipStaticData": {"UserID": IRAN_MMSI, "ImoNumber": 0}},
    }

    assert parse_static(envelope).imo is None


# --------------------------------------------------------------------------
# Subscription frame
# --------------------------------------------------------------------------


def test_subscription_frame_has_required_fields():
    frame = build_subscription("KEY", [[[24.5, 54.0], [27.5, 57.5]]])

    assert frame["APIKey"] == "KEY"
    assert frame["BoundingBoxes"] == [[[24.5, 54.0], [27.5, 57.5]]]
    assert set(frame["FilterMessageTypes"]) == set(SUBSCRIBED_TYPES)


def test_subscription_requires_a_bounding_box():
    """The server rejects a subscription without one."""
    with pytest.raises(ValueError, match="bounding box"):
        build_subscription("KEY", [])


def test_subscription_requires_an_api_key():
    with pytest.raises(ValueError, match="API key"):
        build_subscription("", [[[24.5, 54.0], [27.5, 57.5]]])


def test_subscription_enforces_the_200_mmsi_cap():
    mmsis = [str(400_000_000 + i) for i in range(201)]

    with pytest.raises(ValueError, match="at most 200"):
        build_subscription("KEY", [[[24.5, 54.0], [27.5, 57.5]]], mmsis=mmsis)
