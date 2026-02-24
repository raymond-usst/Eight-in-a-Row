import pytest
import os
import numpy as np
from ai.board_render import board_to_image_path, _HAS_MPL

@pytest.fixture
def temp_img_dir(tmp_path):
    d = tmp_path / "img"
    d.mkdir()
    return str(d)

def test_board_rendering(temp_img_dir):
    if not _HAS_MPL:
        pytest.skip("matplotlib not installed, skipping render test")
        
    board = np.zeros((10, 10), dtype=np.int8)
    
    # Place some pieces
    board[0, 0] = 1 # Red
    board[1, 1] = 2 # Green
    board[2, 2] = 3 # Blue
    board[3, 3] = 4 # Invalid ID, should be safely ignored
    
    out_path = os.path.join(temp_img_dir, "test_render.png")
    success = board_to_image_path(board, out_path, cell_px=10)
    
    assert success is True
    assert os.path.exists(out_path)
    # Basic size check (it shouldn't be an empty file)
    assert os.path.getsize(out_path) > 100

def test_empty_board_input(temp_img_dir):
    if not _HAS_MPL:
        pytest.skip("matplotlib not installed, skipping render test")
        
    out_path = os.path.join(temp_img_dir, "test_empty.png")
    
    # Passing an empty array should fail cleanly
    empty_board = np.array([])
    with pytest.raises(ValueError):
        board_to_image_path(empty_board, out_path)
        
    assert not os.path.exists(out_path)

def test_invalid_board_type(temp_img_dir):
    if not _HAS_MPL:
        pytest.skip("matplotlib not installed, skipping render test")
        
    out_path = os.path.join(temp_img_dir, "test_invalid.png")
    
    with pytest.raises(ValueError):
        board_to_image_path("not_an_array", out_path)
