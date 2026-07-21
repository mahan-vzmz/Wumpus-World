import pytest

from wumpus.domain import Position, Tile
from wumpus.parser import InputFormatError, parse_input


VALID_INPUT = """********
**D*****
*****G**
W***P***
********
********
********
********
100
25
-10
8 8
"""


def test_parser_builds_domain_objects_and_converts_coordinates() -> None:
    parsed = parse_input(VALID_INPUT)
    assert parsed.config.exit_position == Position(7, 7)
    assert parsed.config.pit_score_delta == -10
    assert parsed.game_map.tile_at(Position(1, 2)) is Tile.WALL
    assert parsed.warnings == ()


@pytest.mark.parametrize("exit_text", ["8,8", "(8,8)", "8 8"])
def test_parser_accepts_all_documented_exit_formats(exit_text: str) -> None:
    parsed = parse_input(VALID_INPUT.replace("8 8", exit_text))
    assert parsed.config.exit_position == Position(7, 7)


def test_parser_normalizes_positive_pit_score() -> None:
    parsed = parse_input(VALID_INPUT.replace("-10", "10"))
    assert parsed.config.pit_score_delta == -10
    assert parsed.warnings


def test_parser_allows_pit_on_exit() -> None:
    text = VALID_INPUT.replace("8 8", "4 5")
    parsed = parse_input(text)
    assert parsed.game_map.tile_at(parsed.config.exit_position) is Tile.PIT


def test_parser_rejects_unknown_tile() -> None:
    with pytest.raises(InputFormatError, match="invalid symbol"):
        parse_input(VALID_INPUT.replace("**D*****", "**X*****"))


def test_parser_rejects_dangerous_start() -> None:
    with pytest.raises(InputFormatError, match="start"):
        parse_input(VALID_INPUT.replace("********", "P*******", 1))


def test_parser_rejects_exit_outside_grid() -> None:
    with pytest.raises(InputFormatError, match="inside"):
        parse_input(VALID_INPUT.replace("8 8", "9 9"))


def test_parser_rejects_non_positive_health() -> None:
    with pytest.raises(InputFormatError, match="initial_health"):
        parse_input(VALID_INPUT.replace("100", "0"))