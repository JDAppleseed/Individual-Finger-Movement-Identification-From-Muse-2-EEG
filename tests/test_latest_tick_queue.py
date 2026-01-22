from utils.latest_queue import LatestTickQueue


def test_latest_tick_queue_overwrite():
    queue = LatestTickQueue()
    queue.offer("first")
    queue.offer("second")
    assert queue.try_get_nowait() == "second"
