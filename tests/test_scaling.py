from agentos.scaling import denormalize


def test_origin():
    assert denormalize(0, 0, 1280, 800) == (0, 0)


def test_max_grid_clamps_inside_screen():
    x, y = denormalize(1000, 1000, 1280, 800)
    assert (x, y) == (1279, 799)


def test_midpoint():
    assert denormalize(500, 500, 1280, 800) == (640, 400)


def test_negative_clamps_to_zero():
    assert denormalize(-5, -5, 1280, 800) == (0, 0)


def test_rounding():
    # 333/1000 * 1280 = 426.24 -> 426 ; 667/1000 * 800 = 533.6 -> 534
    assert denormalize(333, 667, 1280, 800) == (426, 534)
