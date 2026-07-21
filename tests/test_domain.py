from wumpus.domain import Action, GameConfig, GameMap, Position, Tile


def test_position_movement_and_neighbors_are_deterministic() -> None:
    position = Position(3, 4)
    assert position.moved(Action.UP) == Position(2, 4)
    assert position.moved(Action.RIGHT) == Position(3, 5)
    assert position.neighbors() == (
        Position(2, 4),
        Position(4, 4),
        Position(3, 3),
        Position(3, 5),
    )


def test_config_derives_default_step_limit() -> None:
    config = GameConfig(100, 25, -10, Position(7, 7))
    assert config.max_steps == 100


def test_map_is_immutable_and_has_eight_by_eight_shape() -> None:
    rows = [[Tile.EMPTY for _ in range(8)] for _ in range(8)]
    game_map = GameMap.from_rows(rows)
    assert game_map.tile_at(Position(0, 0)) is Tile.EMPTY
    assert isinstance(game_map.rows, tuple)
    assert isinstance(game_map.rows[0], tuple)